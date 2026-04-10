"""Kedro pipeline nodes for the ``compute_routes`` pipeline (Model Module 2 — Optional truck-stop dwells).

Inserts *optional* dwell events at public truck-stop locations that lie along
the shortest-path route between each trip's origin and destination.  This
implements the first part of Model Module 2 (Augment Dwell Data): if a
vehicle travels through a truck stop, it could have stopped there to charge,
even if no dwell was recorded in the telematics data.  The pipeline uses a
self-hosted GraphHopper routing engine to compute shortest-path routes, then
performs a spatial join to find truck stops within a buffer of each route.

Pipeline overview
-----------------
1. **import_graph** — Imports an OSM road network into a GraphHopper Docker
   container to prepare it for routing queries.
2. **test_get_routes** — Sends a single cross-country test query to verify
   that the GraphHopper server is healthy before batch routing begins.
3. **filter_routable_trips** — Drops short trips (below
   ``min_dist_miles``) and unnecessary columns; optionally subsamples for
   debugging.
4. **get_trip_orig_dest_points** — Converts origin and destination H3
   hexagons to point geometries and attaches them to the trips GeoDataFrame.
5. **partition_trips** — Re-partitions the Dask GeoDataFrame to the desired
   number of partitions before routing (allows checkpointing to disk).
6. **get_routes_node** — Calls ``get_routes`` partition-by-partition via
   GraphHopper, converting raw metric distances and seconds to miles and
   hours, and setting the route LineString as the active geometry.
7. **format_stop_locations** — Reformats and point-geometrises the truck-stop
   candidate locations (Jason's Law + OSM) for spatial joining.
8. **get_optional_stop_trips** — Spatially joins truck stops within a buffer
   of each route, projects each stop onto the route line to obtain its
   distance from the trip origin, and returns the combined DataFrame of
   original trips plus optional intermediate trips.
9. **describe_optional_stop_trips** — Trims optional stops that are too close
   to the trip endpoints, sorts by trip and distance, and recomputes start/end
   timestamps for each sub-segment using proportional time allocation.
10. **concat_optional_stops** — Appends the optional-stop trips onto the
    original trips DataFrame, deduplicating so that only the split version of
    each affected trip survives.

Key design decisions
--------------------
- **GraphHopper containerised routing**: Running GraphHopper in a Docker
  container on the same machine avoids network latency and rate limits
  associated with hosted routing APIs, which is critical given the ~20 million
  origin-destination pairs in the full dataset.
- **Proportional time allocation**: When a trip is split at an optional stop,
  the time for each sub-segment is computed as
  ``(segment_miles / route_speed) × (observed_hours / route_hours)``
  to preserve the observed start and end timestamps while distributing time
  proportionally to distance.
- **Endpoint buffer exclusion**: Optional stops within ``park_buffer_miles``
  of the trip origin or destination are dropped, as the vehicle would most
  likely have been counted as dwelling there already.
- **Deduplication strategy**: After concatenation, original trips are sorted
  to the bottom (``is_original=False`` sorts before ``True``) and then
  ``drop_duplicates(keep="first")`` on the trip-ID columns retains the
  split version and drops the original.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.

GraphHopper. Open Source Routing Engine. https://www.graphhopper.com/
U.S. DOT Federal Highway Administration. Jason's Law Truck Parking Survey.
"""

import logging

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.diagnostics.progress import ProgressBar
from routingpy import Graphhopper

from laurel.routing.router import (
    DIST_COL,
    ROUTE_COL,
    TIME_COL,
    get_routes,
)
from laurel.routing.server import GraphhopperContainerRouter
from laurel.utils.geo import METERS_PER_MILE
from laurel.utils.h3 import add_geometries, cells_to_points
from laurel.utils.time import SECS_PER_HOUR

logger = logging.getLogger(__name__)


def import_graph(server_params: dict) -> None:
    """Import an OSM road-network file into a GraphHopper Docker container.

    Starts a ``GraphhopperContainerRouter`` in import mode, which instructs
    the container to read the PBF/OSM file and build a routing graph on disk.
    This step only needs to be run once per road-network file; subsequent
    pipeline runs reuse the pre-built graph.

    Args:
        server_params: GraphHopper server configuration dict with keys:

            - ``image`` (str): Docker image name/tag.
            - ``graph_dir`` (str): host path to the directory where the
              routing graph will be stored.
            - ``config_path`` (str): path to the GraphHopper config file.
            - ``map_path`` (str): path to the OSM/PBF input file.
            - ``resources`` (dict): sub-key ``import`` with ``mem_max_gb``,
              ``mem_start_gb``, and ``startup_delay_secs``.
    """
    resource = server_params["resources"]["import"]
    server = GraphhopperContainerRouter(
        image=server_params["image"],
        graph_dir=server_params["graph_dir"],
        config_path=server_params["config_path"],
        mem_max_gb=resource["mem_max_gb"],
        mem_start_gb=resource["mem_start_gb"],
        startup_delay=resource["startup_delay_secs"],
    )
    server.import_graph(input_file=server_params["map_path"])
    logger.info("Import completed")


def test_get_routes(route_params: dict, server_params: dict) -> None:
    """Send a health-check routing query to verify the GraphHopper server is operational.

    Issues a single coast-to-coast route request (Vermont to California) and
    logs the returned distance.  The pipeline should be halted if this node
    fails, as it indicates the routing server is unavailable.

    Args:
        route_params: Route configuration dict with keys:

            - ``profile`` (str): GraphHopper vehicle profile (e.g.,
              ``"car"`` or ``"truck"``).
        server_params: GraphHopper server configuration dict (same structure
            as ``import_graph``); the ``resources.server`` sub-key is used.
    """
    coords = [(-72.21865, 43.73610), (-122.15615, 37.42383)]  # A cross-US route
    resource = server_params["resources"]["server"]
    with GraphhopperContainerRouter(
        image=server_params["image"],
        graph_dir=server_params["graph_dir"],
        config_path=server_params["config_path"],
        mem_max_gb=resource["mem_max_gb"],
        mem_start_gb=resource["mem_start_gb"],
        startup_delay=resource["startup_delay_secs"],
    ) as server:
        router = Graphhopper(base_url=server.base_url)
        route = router.directions(locations=coords, profile=route_params["profile"])
        logger.info(f"Route distance: {route.distance} meters")


def filter_routable_trips(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Drop unnecessary columns and short trips that are not worth routing.

    Very short trips (below ``min_dist_miles``) are excluded because routing
    them would add noise without meaningfully changing the set of reachable
    truck stops.  An optional debug subsample further reduces the dataset for
    rapid iteration.

    Args:
        trips: Dask DataFrame of formatted trip records.
        params: Pipeline parameters dict with keys:

            - ``drop_cols`` (list[str]): columns to remove before routing.
            - ``dist_col`` (str): trip-distance column name (miles).
            - ``min_dist_miles`` (float): minimum trip distance to retain.
            - ``debug_subsample`` (dict): ``active`` (bool) and ``frac``
              (float) for fractional subsampling.

    Returns:
        A filtered Dask DataFrame retaining only trips long enough to route.
    """
    trips = trips.drop(columns=params["drop_cols"])
    long_enough_trip = trips[params["dist_col"]] >= params["min_dist_miles"]
    trips = trips.loc[long_enough_trip]
    if params["debug_subsample"]["active"]:
        trips = trips.sample(frac=params["debug_subsample"]["frac"])
    return trips


def get_trip_orig_dest_points(trips: dd.DataFrame, params: dict) -> dgpd.GeoDataFrame:
    """Convert origin and destination H3 hexagons to point geometries for routing.

    Maps each H3 hex-ID column to a ``GeoSeries`` of centroid points using
    ``cells_to_points``, then sets the active geometry to the output geometry
    column.

    Args:
        trips: Filtered trips Dask DataFrame with H3 hex-ID columns.
        params: Pipeline parameters dict with keys:

            - ``hex_geo_cols`` (dict[str, str]): mapping from output geometry
              column name to the source hex-ID column name (e.g.,
              ``{"origin_geom": "origin_hex"}``).
            - ``output_geom_col`` (str): name of the active geometry column
              to set on the output GeoDataFrame.

    Returns:
        A Dask GeoDataFrame with point geometry columns for origin and
        destination.
    """
    trips = dgpd.from_dask_dataframe(trips, geometry=None)
    for tgt, src in params["hex_geo_cols"].items():
        trips[tgt] = trips[src].map_partitions(cells_to_points, meta=gpd.GeoSeries())
    trips = trips.set_geometry(params["output_geom_col"])
    return trips


def partition_trips(trips: dgpd.GeoDataFrame, params: dict) -> dgpd.GeoDataFrame:
    """Re-partition the trips GeoDataFrame before writing to disk.

    Args:
        trips: Trips Dask GeoDataFrame with origin/destination geometries.
        params: Pipeline parameters dict with keys:

            - ``n_partitions`` (int): target number of Dask partitions.

    Returns:
        The GeoDataFrame repartitioned to ``params["n_partitions"]`` parts.
    """
    parts = trips.repartition(npartitions=params["n_partitions"])
    return parts


def get_routes_node(
    trips: dgpd.GeoDataFrame,
    server: GraphhopperContainerRouter,
    params: dict,
) -> dgpd.GeoDataFrame:
    """Compute shortest-path routes for all trips and convert units to miles and hours.

    Calls ``get_routes`` on each Dask partition via ``map_partitions``,
    forwarding the origin and destination geometry columns and GraphHopper
    client parameters.  After routing, raw metric units are converted:
    distance from metres to miles, duration from seconds to hours, and speed
    is derived as miles per hour.  The route LineString column becomes the
    active geometry.

    Args:
        trips: Trips Dask GeoDataFrame with origin/destination point columns
            (output of ``partition_trips``).
        server: Running ``GraphhopperContainerRouter`` context manager
            providing ``server.base_url``.
        params: Pipeline parameters dict with keys:

            - ``input_cols`` (dict): sub-keys ``orig`` and ``dest`` naming
              the origin and destination geometry columns.
            - ``client`` (dict): ``max_concurrent_requests``, ``batch_size``,
              ``timeout_secs``, ``verbose`` — forwarded to ``get_routes``.
            - ``profile`` (str): GraphHopper vehicle profile.
            - ``output_trip_cols`` (dict): sub-keys ``dist``, ``dur``,
              ``speed`` naming the output columns.

    Returns:
        A Dask GeoDataFrame with route LineString geometry and added columns
        for route distance (miles), duration (hours), and speed (mph).
    """
    logger.info("Starting routing")
    icols = params["input_cols"]

    routed = trips.map_partitions(
        get_routes,
        orig_col=icols["orig"],
        dest_col=icols["dest"],
        max_concurrent_requests=params["client"]["max_concurrent_requests"],
        batch_size=params["client"]["batch_size"],
        timeout=params["client"]["timeout_secs"],
        verbose=params["client"]["verbose"],
        server_url=server.base_url,
        profile=params["profile"],
    )
    logger.info("Finished routing")

    logger.info("Interpreting routes")
    tcols = params["output_trip_cols"]
    routed[tcols["dist"]] = routed[DIST_COL] / METERS_PER_MILE
    routed[tcols["dur"]] = routed[TIME_COL] / SECS_PER_HOUR
    routed[tcols["speed"]] = routed[tcols["dist"]] / routed[tcols["dur"]]
    routed = routed.drop(columns=[DIST_COL, TIME_COL])
    routed = routed.set_geometry(ROUTE_COL)
    return routed


def format_stop_locations(stops: pd.DataFrame, params: dict) -> gpd.GeoDataFrame:
    """Convert truck-stop candidate records to a point GeoDataFrame for spatial joining.

    Resets the index, applies column renames, creates point geometries from
    the H3 hexagon centroids, renames the geometry column, and assigns a
    contiguous integer stop ID.

    Args:
        stops: Raw truck-stop DataFrame (Jason's Law or similar) indexed by
            hexagon ID.
        params: Pipeline parameters dict with keys:

            - ``columns`` (dict): sub-keys ``hex`` (hex-ID column),
              ``park_point`` (output geometry column name), ``park_id``
              (output stop-ID column name).
            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names (inverted for renaming).
            - ``keep_cols`` (list[str]): columns to retain in the output.

    Returns:
        A ``gpd.GeoDataFrame`` with one row per truck-stop candidate and a
        point geometry column named ``params["columns"]["park_point"]``.
    """
    pcols = params["columns"]
    stops_ren = stops.reset_index()
    stops_ren = stops_ren.rename(
        columns={v: k for k, v in params["col_renamer"].items()}
    )
    stops_geo = add_geometries(stops_ren, hex_col=pcols["hex"], geom_type="point")
    stops_geo = stops_geo.rename_geometry(pcols["park_point"])
    stops_geo[pcols["park_id"]] = pd.RangeIndex(stop=stops_geo.shape[0])
    stops_out = stops_geo.loc[:, params["keep_cols"]]
    return stops_out


def get_optional_stop_trips(
    routes: dgpd.GeoDataFrame, parks: gpd.GeoDataFrame, params: dict
) -> pd.DataFrame:
    """Identify truck stops along each route and compute their distance from the trip origin.

    The procedure is:

    1. Buffer each truck-stop point by ``park_buffer_miles`` in a projected
       CRS to create a catchment polygon.
    2. Drop routed trips with null geometry; spatially join the buffered stop
       polygons onto the route LineStrings using ``intersects``.
    3. For each matched (trip, stop) pair, project the stop-point geometry
       onto the route LineString to obtain its distance from the trip origin
       (in miles).
    4. Append the optional-stop rows to the original trips rows (with
       ``is_optional`` flags) and trigger a Dask compute with a progress bar.

    Route geometries are dropped from the optional-stop rows after projection
    to reduce memory usage.

    Args:
        routes: Routed trips Dask GeoDataFrame with route LineString geometry.
        parks: Truck-stop GeoDataFrame (output of ``format_stop_locations``).
        params: Pipeline parameters dict with keys:

            - ``columns`` (dict): sub-keys for column names including
              ``route_geom``, ``park_point``, ``park_id``, ``hex_end``,
              ``hex_park``, ``dist_along_miles``.
            - ``projected_crs`` (str | CRS): CRS used for buffering and
              distance projection.
            - ``park_buffer_miles`` (float): buffer radius around each truck
              stop (miles).
            - ``drop_cols_initial`` (list[str]): columns to drop from the
              routes DataFrame before joining.
            - ``progress_report_interval_secs`` (float): Dask progress bar
              reporting interval.

    Returns:
        An in-memory ``pd.DataFrame`` combining original trips
        (``is_optional=False``) and optional truck-stop trips
        (``is_optional=True``), with a ``dist_along_miles`` column recording
        each record's distance from the trip origin.
    """
    pcols = params["columns"]

    # Set up the parks for spatial join
    parks = parks.to_crs(params["projected_crs"])
    parks["buffer"] = parks.geometry.buffer(
        distance=params["park_buffer_miles"] * METERS_PER_MILE
    )
    parks = parks.set_geometry("buffer")

    # Eliminate routes with no geometry and unused columns
    trips_source = routes.dropna(subset=[pcols["route_geom"]])
    trips_source = trips_source.drop(columns=params["drop_cols_initial"])

    # Spatial join
    trips_short = trips_source.to_crs(params["projected_crs"])
    trips_short = trips_short.sjoin(parks, how="inner", predicate="intersects")
    trips_short = trips_short.drop(columns=["index_right"])

    # Find distances along the route for each optional stop
    def project_partition(
        part: gpd.GeoDataFrame, line_col: str, point_col: str, out_col: str
    ) -> gpd.GeoDataFrame:
        """Project the points_col of the partition on to the line_col."""
        part[out_col] = part[line_col].project(part[point_col]) / METERS_PER_MILE
        return part

    trips_short[pcols["dist_along_miles"]] = np.nan
    trips_short = trips_short.map_partitions(
        project_partition,
        line_col=pcols["route_geom"],
        point_col=pcols["park_point"],
        out_col=pcols["dist_along_miles"],
        meta=trips_short,
    )

    # After this point, the route geometries are no longer needed, so we drop them to save memory
    trips_short[pcols["hex_end"]] = trips_short[pcols["hex_park"]]
    trips_short = trips_short.drop(
        columns=[pcols["route_geom"], pcols["park_point"], pcols["hex_park"]]
    )
    trips_short["is_optional"] = True

    # Prepare the original trips for concatenation
    trips_source["dist_along_miles"] = trips_source["trip_miles_route"]
    trips_orig = trips_source.drop(columns=pcols["route_geom"])
    trips_orig["is_optional"] = False

    # Concatenate trips
    trips_mod = dd.concat([trips_orig, trips_short], axis=0)

    logger.info("Computing the optional stop trips by spatial joining and projecting.")
    with ProgressBar(dt=params["progress_report_interval_secs"]):
        trips_mod = trips_mod.compute()
    return trips_mod


def describe_optional_stop_trips(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Compute split timestamps and distances for optional-stop sub-trips.

    For each trip that has been split at one or more truck stops, this
    function:

    1. Drops optional stops within ``park_buffer_miles`` of the trip origin or
       destination (where the vehicle is likely already counted as dwelling).
    2. Sorts all rows (original + optional) by trip ID and distance from
       origin to establish sub-trip ordering.
    3. Computes each sub-segment's distance as ``dist_along - dist_prev`` and
       its duration as ``(seg_miles / route_speed) × (obs_hours / route_hours)``
       (proportional scaling preserves the observed total trip time).
    4. Derives cumulative time shifts from the trip start time to produce new
       ``start_time`` and ``end_time`` for each sub-trip, rounded to seconds.
    5. Renames and selects columns to match the original trips schema.

    Args:
        trips: Combined DataFrame of original and optional-stop trips (output
            of ``get_optional_stop_trips``).
        params: Pipeline parameters dict with keys:

            - ``columns`` (dict): sub-keys for ``dist_along_miles``,
              ``miles_route``, ``is_optional``, ``speed_route``,
              ``hours_orig``, ``hours_route``, ``start_time``.
            - ``park_buffer_miles`` (float): buffer distance used to exclude
              endpoint-adjacent stops.
            - ``trip_id_cols`` (list[str]): columns that uniquely identify a
              trip (used for groupby and sort).
            - ``rename_cols_final`` (dict[str, str]): column renames applied
              at the end to restore original column names.
            - ``keep_cols_final`` (list[str]): columns to retain in the output.

    Returns:
        A ``pd.DataFrame`` of sub-trips with updated timestamps and distances,
        ready to be concatenated with the original trips.
    """
    pcols = params["columns"]

    # Eliminate optional trips too close to the ends of the original trip
    started_at_park = trips[pcols["dist_along_miles"]] < params["park_buffer_miles"]
    ended_at_park = trips[pcols["dist_along_miles"]] > (
        trips[pcols["miles_route"]] - params["park_buffer_miles"]
    )
    is_opt = trips[pcols["is_optional"]]
    trips = trips.loc[(~started_at_park & ~ended_at_park & is_opt) | ~is_opt, :]

    logger.info("Sort trips to enable position-based computations")
    trip_id_cols = params["trip_id_cols"]
    trips = trips.sort_values(
        trip_id_cols + [pcols["dist_along_miles"]], ascending=True
    )

    logger.info("Compute new timings and distances for optional and original trips")
    # Distances by segment
    trips["dist_prev_miles"] = trips.groupby(trip_id_cols)[
        pcols["dist_along_miles"]
    ].shift(1, fill_value=0.0)
    trips["trip_miles_route_seg"] = (
        trips[pcols["dist_along_miles"]] - trips["dist_prev_miles"]
    )

    # Times by segment
    trips["trip_hrs_route_seg"] = (
        trips["trip_miles_route_seg"] / trips[pcols["speed_route"]]
    )
    time_scaler = trips[pcols["hours_orig"]] / trips[pcols["hours_route"]]
    trips["trip_hrs_route_seg"] = trips["trip_hrs_route_seg"] * time_scaler
    trips["trip_time_route_seg"] = pd.to_timedelta(
        trips["trip_hrs_route_seg"], unit="h"
    )
    trips["time_shift"] = trips.groupby(trip_id_cols)["trip_time_route_seg"].cumsum()

    trips["new_end"] = trips[pcols["start_time"]] + trips["time_shift"]
    trips["new_end"] = trips["new_end"].dt.round("s")
    trips["new_start"] = trips.groupby(trip_id_cols)["new_end"].shift(1)
    trips["new_start"] = trips["new_start"].fillna(trips[pcols["start_time"]])

    # Format to match original trips dataset
    drop_col_set = set(params["rename_cols_final"].keys())
    drop_col_set = drop_col_set.intersection(trips.columns)
    trips = trips.drop(columns=drop_col_set)
    trips_out = trips.rename(
        columns={v: k for k, v in params["rename_cols_final"].items()}
    )
    trips_out = trips_out.loc[:, params["keep_cols_final"]]
    return trips_out


def concat_optional_stops(
    trips_orig: dd.DataFrame, trips_opt: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Merge optional-stop sub-trips with the original trips, replacing split trips.

    Concatenates the original Dask trips (materialised in memory) with the
    optional-stop sub-trip records.  For trips that were split at truck stops,
    the original trip row must be replaced by the split sub-trips.  This is
    achieved by:

    1. Flagging original trips as ``is_original=True`` and optional trips as
       ``is_original=False``.
    2. Sorting by trip-ID columns followed by ``is_original`` ascending, so
       optional (split) rows sort before the original row for the same trip.
    3. Calling ``drop_duplicates(subset=trip_id_cols, keep="first")`` to
       retain the optional rows and discard the originals.

    The resulting pandas DataFrame is wrapped back into a Dask DataFrame for
    downstream pipeline compatibility.

    Args:
        trips_orig: Original trips Dask DataFrame (pre-routing).
        trips_opt: Optional-stop sub-trips (output of
            ``describe_optional_stop_trips``).
        params: Pipeline parameters dict with keys:

            - ``drop_cols`` (list[str]): columns to drop from ``trips_orig``
              before concatenation.
            - ``trip_id_cols`` (list[str]): columns uniquely identifying a
              trip row.
            - ``n_partitions`` (int): number of Dask partitions for the
              output DataFrame.

    Returns:
        A Dask DataFrame combining original unmodified trips and split
        optional-stop sub-trips, with one row per unique (trip_id_cols)
        combination.
    """
    logger.info("Computing original trips into memory.")
    trips_orig = trips_orig.drop(columns=params["drop_cols"])
    trips_orig = trips_orig.compute()

    logger.info("Concatenating and sorting trips.")
    trips_orig["is_original"] = True
    trips_opt["is_original"] = False
    trips = pd.concat([trips_orig, trips_opt], axis=0)

    # Drop the original versions of trips which have been split
    sort_cols = params["trip_id_cols"] + ["is_original"]
    trips = trips.sort_values(sort_cols, ascending=True)
    # We keep the first trip because we want to replace original with modified trips
    trips = trips.drop_duplicates(subset=params["trip_id_cols"], keep="first")
    trips = trips.drop(columns=["is_original"])

    # Send back to Dask for later processing
    trips_out = dd.from_pandas(trips, npartitions=params["n_partitions"])
    return trips_out
