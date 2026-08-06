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
8. **get_optional_stop_trips** — Per partition, spatially joins truck stops
   within a buffer of each route, projects each stop onto the route line to
   obtain its distance from the trip origin, drops optional stops too close
   to either trip endpoint, and returns a lazy Dask DataFrame combining
   original trips with the surviving optional intermediate trips.
9. **describe_optional_stop_trips** — Per partition, sorts by trip and
   distance and recomputes start/end timestamps for each sub-segment using
   proportional time allocation.
10. **concat_optional_stops** — Anti-joins the optional-stop trips against the
    original trips DataFrame to drop the original (unsplit) row for any trip
    that was split, concatenates what remains, and sorts/indexes the result
    by trip ID.

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
- **Deduplication strategy**: A left anti-join drops any original trip row
  whose trip ID appears among the optional-stop sub-trips, so the split
  version replaces the original rather than coexisting with it.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.

GraphHopper. Open Source Routing Engine. https://www.graphhopper.com/
U.S. DOT Federal Highway Administration. Jason's Law Truck Parking Survey.
"""

from __future__ import annotations

import logging

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
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


def import_graph(osm_path: str, server_params: dict) -> None:
    """Import an OSM road-network file into a GraphHopper Docker container.

    Starts a ``GraphhopperContainerRouter`` in import mode, which instructs
    the container to read the PBF/OSM file and build a routing graph on disk.
    This step only needs to be run once per road-network file; subsequent
    pipeline runs reuse the pre-built graph.

    Args:
        osm_path: Path to the OSM/PBF input file, supplied by the
            ``osm_north_america`` catalog entry. It arrives fully resolved, which
            matters because the container gets it verbatim and cannot expand
            environment variables such as ``$SCRATCH``.
        server_params: GraphHopper server configuration dict with keys:

            - ``image`` (str): Docker image name/tag.
            - ``graph_dir`` (str): host path to the directory where the
              routing graph will be stored.
            - ``config_path`` (str): path to the GraphHopper config file.
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
    server.import_graph(input_file=osm_path)
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


def _build_partition_trips(
    part: gpd.GeoDataFrame,
    parks: gpd.GeoDataFrame,
    params: dict,
) -> pd.DataFrame:
    """Compute one partition's full, filtered contribution: its own trip
    row(s) (``is_optional=False``) plus any optional-stop split rows from
    joining against ``parks`` (``is_optional=True``), with endpoint-adjacent
    stops already excluded (``describe_optional_stop_trips`` still does its
    own sort, right before the position-based math that needs it).

    Runs as one task per partition -- no ``dask_geopandas`` broadcast-join
    graph layer inside this function, so the optimizer can fuse it directly
    onto the upstream read. ``part`` is never mutated in place; every
    transformation below produces a new frame, so the original routed-trip
    data always remains available to both branches independently.

    Valid only because every trip's original row and all of its
    optional-stop splits are guaranteed to live in the same partition of
    ``routes`` (``partition_trips`` repartitions before any join can
    multiply rows). Do not introduce a ``.repartition()``/``.set_index()``/
    shuffle anywhere between ``partition_trips`` and
    ``describe_optional_stop_trips`` without re-verifying this invariant.
    """
    pcols = params["columns"]

    trips_source = part.dropna(subset=[pcols["route_geom"]]).drop(
        columns=params["drop_cols_initial"]
    )
    del part

    orig = pd.DataFrame(trips_source.drop(columns=[pcols["route_geom"]]))
    orig[pcols["dist_along_miles"]] = orig["trip_miles_route"]
    orig["is_optional"] = False

    part_proj = trips_source.set_geometry(pcols["route_geom"]).to_crs(
        params["projected_crs"]
    )
    del trips_source

    short = gpd.sjoin(part_proj, parks, how="inner", predicate="intersects")
    del part_proj
    short = short.drop(columns=["index_right"])
    short[pcols["dist_along_miles"]] = (
        short[pcols["route_geom"]].project(short[pcols["park_point"]])
        / METERS_PER_MILE
    )
    short[pcols["hex_end"]] = short[pcols["hex_park"]]
    short = pd.DataFrame(
        short.drop(columns=[pcols["route_geom"], pcols["park_point"], pcols["hex_park"]])
    )
    short["is_optional"] = True

    trips = pd.concat([orig, short], axis=0)
    del orig, short

    # Drop optional stops too close to either trip endpoint (already
    # counted as a dwell there), then sort for the segment distance/time
    # math below.
    started_at_park = trips[pcols["dist_along_miles"]] < params["park_buffer_miles"]
    ended_at_park = trips[pcols["dist_along_miles"]] > (
        trips["trip_miles_route"] - params["park_buffer_miles"]
    )
    is_opt = trips[pcols["is_optional"]]
    trips = trips.loc[(~started_at_park & ~ended_at_park & is_opt) | ~is_opt, :]
    return trips


def get_optional_stop_trips(
    routes: dgpd.GeoDataFrame, parks: gpd.GeoDataFrame, params: dict
) -> dd.DataFrame:
    """Identify truck stops along each route and compute their distance from the trip origin.

    Fuses, into a single per-partition task, everything that used to be a
    chain of separate Dask operations: dropping null-geometry routes and
    unused columns, buffering and spatially joining truck stops onto the
    route LineStrings, projecting each matched stop onto its route to get
    its distance from the trip origin, dropping the now-unneeded route
    geometry, and excluding optional stops too close to either trip
    endpoint. Returns a lazy ``dd.DataFrame`` -- nothing is computed here;
    the pipeline's only materialization point is the final catalog write.

    Args:
        routes: Routed trips Dask GeoDataFrame with route LineString geometry.
        parks: Truck-stop GeoDataFrame (output of ``format_stop_locations``).
        params: Pipeline parameters dict with keys:

            - ``columns`` (dict): sub-keys for column names including
              ``route_geom``, ``park_point``, ``park_id``, ``hex_end``,
              ``hex_park``, ``dist_along_miles``, ``is_optional``.
            - ``projected_crs`` (str | CRS): CRS used for buffering and
              distance projection.
            - ``park_buffer_miles`` (float): buffer radius around each truck
              stop (miles); also used to exclude stops adjacent to either
              trip endpoint.
            - ``drop_cols_initial`` (list[str]): columns to drop from the
              routes DataFrame before joining.

    Returns:
        A lazy ``dd.DataFrame`` combining original trips
        (``is_optional=False``) and optional truck-stop trips
        (``is_optional=True``), with a ``dist_along_miles`` column recording
        each record's distance from the trip origin.
    """
    parks = parks.to_crs(params["projected_crs"])
    parks["buffer"] = parks.geometry.buffer(
        distance=params["park_buffer_miles"] * METERS_PER_MILE
    )
    parks = parks.set_geometry("buffer")

    # Broadcast as a 1-partition collection so `parks` is a shared
    # dependency, not re-embedded as a literal in every partition task.
    parks_dgpd = dgpd.from_geopandas(parks, npartitions=1)

    meta = _build_partition_trips(routes._meta_nonempty, parks, params)
    return routes.map_partitions(_build_partition_trips, parks_dgpd, params, meta=meta)


def _describe_partition(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Compute split timestamps and distances for one partition's sub-trips.

    Endpoint-adjacent optional stops are already excluded by
    ``_build_partition_trips``; this sorts by trip ID and distance from
    origin to establish sub-trip ordering, computes each sub-segment's
    distance as ``dist_along - dist_prev`` and its duration as
    ``(seg_miles / route_speed) x (obs_hours / route_hours)`` (proportional
    scaling preserves the observed total trip time), derives cumulative time
    shifts from the trip start time to produce new ``start_time``/``end_time``
    for each sub-trip rounded to seconds, then renames/selects columns to
    match the original trips schema.
    """
    pcols = params["columns"]
    trip_id_cols = params["trip_id_cols"]

    trips = trips.sort_values(trip_id_cols + [pcols["dist_along_miles"]], ascending=True)

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
    return trips_out.loc[:, params["keep_cols_final"]]


def describe_optional_stop_trips(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Compute split timestamps and distances for optional-stop sub-trips.

    Applies ``_describe_partition`` lazily to each partition of ``trips``
    (output of ``get_optional_stop_trips``). Safe to run per-partition
    because every trip's rows are guaranteed to live in one partition (see
    ``_build_partition_trips``'s docstring).

    Args:
        trips: Combined Dask DataFrame of original and optional-stop trips
            (output of ``get_optional_stop_trips``).
        params: Pipeline parameters dict with keys:

            - ``columns`` (dict): sub-keys for ``dist_along_miles``,
              ``speed_route``, ``hours_orig``, ``hours_route``, ``start_time``.
            - ``trip_id_cols`` (list[str]): columns that uniquely identify a
              trip (used for groupby and sort).
            - ``rename_cols_final`` (dict[str, str]): column renames applied
              at the end to restore original column names.
            - ``keep_cols_final`` (list[str]): columns to retain in the output.

    Returns:
        A lazy ``dd.DataFrame`` of sub-trips with updated timestamps and
        distances, ready to be concatenated with the original trips.
    """
    meta = _describe_partition(trips._meta_nonempty, params)
    return trips.map_partitions(_describe_partition, params, meta=meta)


def concat_optional_stops(
    trips_orig: dd.DataFrame, trips_opt: dd.DataFrame, params: dict
) -> dd.DataFrame:
    """Merge optional-stop sub-trips with the original trips, replacing split trips.

    For trips that were split at truck stops, the original trip row must be
    replaced by the split sub-trips. Rather than gathering everything into
    one process to sort-and-dedup, this does a left anti-join to drop
    exactly the original rows whose trip ID appears in ``trips_opt``, then
    concatenates what's left with ``trips_opt`` -- both Dask-native,
    shuffle-based operations rather than a full gather.

    The result is sorted by ``trip_id_cols`` and indexed by the leading
    trip-ID column (``veh_id``): ``set_index`` shuffles once to guarantee
    every vehicle's rows land in a single partition and are ordered across
    partitions, then a cheap per-partition sort handles the remaining
    trip-ID column(s) within each vehicle's rows -- avoiding a full
    multi-key global sort.

    Args:
        trips_orig: Original trips Dask DataFrame (pre-routing).
        trips_opt: Optional-stop sub-trips (output of
            ``describe_optional_stop_trips``).
        params: Pipeline parameters dict with keys:

            - ``drop_cols`` (list[str]): columns to drop from ``trips_orig``
              before concatenation.
            - ``trip_id_cols`` (list[str]): columns uniquely identifying a
              trip row; the first is used as the output index.
            - ``n_partitions`` (int): output partition count, applied as
              part of the ``set_index`` shuffle below.

    Returns:
        A Dask DataFrame combining original unmodified trips and split
        optional-stop sub-trips, with one row per unique (trip_id_cols)
        combination, indexed by the leading ``trip_id_cols`` entry.
    """
    trips_orig = trips_orig.drop(columns=params["drop_cols"])

    opt_ids = trips_opt[params["trip_id_cols"]].drop_duplicates()
    merged = trips_orig.merge(
        opt_ids, on=params["trip_id_cols"], how="left", indicator=True
    )
    trips_orig_kept = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    del merged

    trips = dd.concat([trips_orig_kept, trips_opt], axis=0)
    del trips_orig_kept

    id_col, *rest_id_cols = params["trip_id_cols"]
    trips = trips.set_index(id_col, sorted=False, npartitions=params["n_partitions"])
    if rest_id_cols:
        # Stable sort by the remaining ID columns first, then a stable
        # sort_index: ties (same veh_id) keep their rest_id_cols order from
        # the first pass, while the index stays monotonic per partition --
        # required for `.loc[scalar]` lookups on the duplicate-valued index.
        trips = trips.map_partitions(
            lambda df: df.sort_values(rest_id_cols, kind="stable").sort_index(
                kind="stable"
            )
        )
    return trips
