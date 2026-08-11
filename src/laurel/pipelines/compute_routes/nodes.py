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
3. **select_trips_to_route** — Drops unnecessary columns and splits trips
   into those worth routing and those below ``min_dist_miles``; optionally
   subsamples the to-route side for debugging.
4. **index_by_vehicle** — Indexes a trips DataFrame by vehicle ID with known
   divisions. Run once, on the full trips set before
   ``select_trips_to_route``, so both the to-route and not-to-route sides
   inherit it for free, and it survives unchanged all the way through
   ``describe_optional_stop_trips``.
5. **get_trip_orig_dest_points** — Converts origin and destination H3
   hexagons to point geometries and attaches them to the trips GeoDataFrame.
6. **partition_trips** — Re-partitions the Dask GeoDataFrame to the desired
   number of partitions before routing (allows checkpointing to disk).
7. **get_routes_node** — Calls ``get_routes`` partition-by-partition via
   GraphHopper, converting raw metric distances and seconds to miles and
   hours, and setting the route LineString as the active geometry.
8. **format_stop_locations** — Reformats and point-geometrises the truck-stop
   candidate locations (Jason's Law + OSM) for spatial joining.
9. **get_optional_stop_trips** — Per partition, spatially joins truck stops
   within a buffer of each route, projects each stop onto the route line to
   obtain its distance from the trip origin, drops optional stops too close
   to either trip endpoint, and returns a lazy Dask DataFrame combining
   original trips with the surviving optional intermediate trips.
10. **describe_optional_stop_trips** — Per partition, sorts by trip and
    distance and recomputes start/end timestamps for each sub-segment using
    proportional time allocation.
11. **concat_optional_stops** — Combines the never-routed trips with the
    described optional-stop trips via a divisions-aware ``dd.concat``, then
    sorts the result by trip ID.

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
- **Deduplication by construction, not anti-join**: trips too short to route
  and routed trips are disjoint sets from the moment ``select_trips_to_route``
  splits them -- a trip can only ever appear in one of the two collections
  ``concat_optional_stops`` combines, so no anti-join is needed to avoid
  double-counting split trips.
- **Divisions survive the checkpoint, not a second shuffle**: the vehicle-ID
  index and its divisions are established once (``index_by_vehicle``, before
  ``select_trips_to_route``) and never shuffled again -- every node between
  there and ``describe_optional_stop_trips`` operates per-partition only
  (see ``_build_partition_trips``'s no-shuffle invariant). The catalog entry
  ``describe_optional_stop_trips`` writes to is loaded back with
  ``calculate_divisions: True``, which recovers known divisions from the
  written partitions' index ranges instead of paying for an unsorted
  ``set_index`` (divisions-sampling pass, then data-transfer pass) a second
  time.

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


def select_trips_to_route(
    trips: dd.DataFrame, params: dict
) -> tuple[dd.DataFrame, dd.DataFrame]:
    """Split trips into those worth routing and those too short to bother.

    Very short trips (below ``min_dist_miles``) are excluded from routing
    because it would add noise without meaningfully changing the set of
    reachable truck stops -- they can never gain an optional stop, so they
    are returned separately rather than dropped, ready to be reattached
    unchanged by ``concat_optional_stops`` further downstream. An optional
    debug subsample further reduces the to-route set for rapid iteration;
    it is not applied to the not-to-route set, since that set isn't part of
    the expensive routing path debug_subsample exists to shrink.

    ``trips`` is expected to already be indexed by vehicle ID (see
    ``index_by_vehicle``) -- both returned frames inherit that index and its
    divisions for free via ``.loc[]``, so nothing downstream needs to
    re-index the not-to-route side.

    Args:
        trips: Dask DataFrame of formatted trip records, indexed by vehicle
            ID.
        params: Pipeline parameters dict with keys:

            - ``drop_cols`` (list[str]): columns to remove before routing.
            - ``dist_col`` (str): trip-distance column name (miles).
            - ``min_dist_miles`` (float): minimum trip distance to retain.
            - ``debug_subsample`` (dict): ``active`` (bool) and ``frac``
              (float) for fractional subsampling.

    Returns:
        A ``(to_route, not_to_route)`` tuple of Dask DataFrames.
    """
    trips = trips.drop(columns=params["drop_cols"])
    long_enough_trip = trips[params["dist_col"]] >= params["min_dist_miles"]
    to_route = trips.loc[long_enough_trip]
    not_to_route = trips.loc[~long_enough_trip]
    if params["debug_subsample"]["active"]:
        to_route = to_route.sample(frac=params["debug_subsample"]["frac"])
    return to_route, not_to_route


def index_by_vehicle(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Index a trips DataFrame by vehicle ID with known, monotonic divisions.

    Used at two points in this pipeline where the caller needs a collection
    with real divisions on the vehicle-ID column so it can later be combined
    with another such collection via ``dd.concat(..., interleave_partitions=True)``
    -- a partition-wise merge rather than a full hash shuffle. An unsorted
    ``set_index`` does its own divisions-sampling pass and its own
    data-transfer pass over its input, so this is only called on inputs that
    are either cheap (no expensive ancestry to redo twice) or already
    checkpointed to disk (severing any expensive ancestry beforehand).

    Args:
        trips: Dask DataFrame to index.
        params: Pipeline parameters dict with keys:

            - ``id_col`` (str): vehicle-ID column to index by.
            - ``n_partitions`` (int): target output partition count.

    Returns:
        ``trips`` indexed by ``id_col``, with known divisions.
    """
    return trips.set_index(params["id_col"], sorted=False, npartitions=params["n_partitions"])


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
    row(s) (``routing_status="original"`` or ``"unrouted"``) plus any
    optional-stop split rows from joining against ``parks``
    (``routing_status="routed"``), with endpoint-adjacent stops already
    excluded (``describe_optional_stop_trips`` still does its own sort,
    right before the position-based math that needs it).

    A trip with no route geometry is tagged ``"unrouted"`` instead of being
    dropped, and contributes only its own row (no route path exists to
    check against truck stops) -- ``_describe_partition`` fills in its
    distance/time from pre-routing ``trip_miles``/``trip_hrs`` before doing
    its segment math. This covers two distinct cases that
    ``laurel.routing.router`` cannot tell apart once a trip reaches this
    function: the trip's origin and destination H3 hexes are identical
    (``router``'s ``orig == dest`` short-circuit -- which does not imply
    zero distance traveled, since hex centroids rather than raw GPS points
    are being compared), or GraphHopper genuinely failed to route it.

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

    trips_source = part.drop(columns=params["drop_cols_initial"])
    del part

    no_route = trips_source[pcols["route_geom"]].isna()

    # Get the original routed trips
    routed_source = trips_source.loc[~no_route]

    orig = pd.DataFrame(routed_source.drop(columns=[pcols["route_geom"]]))
    orig[pcols["dist_along_miles"]] = orig["trip_miles_route"]
    orig[pcols["routing_status"]] = "original"

    # Mark trips that did not get routes for later fallback behavior
    unrouted = pd.DataFrame(
        trips_source.loc[no_route].drop(columns=[pcols["route_geom"]])
    )
    unrouted[pcols["routing_status"]] = "unrouted"

    # Identify optional stops on routed trips
    part_proj = routed_source.set_geometry(pcols["route_geom"]).to_crs(
        params["projected_crs"]
    )
    del routed_source

    short = gpd.sjoin(part_proj, parks, how="inner", predicate="intersects")
    del part_proj
    short = short.drop(columns=["index_right"])
    short[pcols["dist_along_miles"]] = (
        short[pcols["route_geom"]].project(short[pcols["park_point"]])
        / METERS_PER_MILE
    )
    started_at_park = short[pcols["dist_along_miles"]] < params["park_buffer_miles"]
    ended_at_park = short[pcols["dist_along_miles"]] > (
        short["trip_miles_route"] - params["park_buffer_miles"]
    )
    short = short.loc[(~started_at_park & ~ended_at_park), :]
    # The truck stop becomes this split's new endpoint, so the trip's
    # existing hex_end must go before the rename -- otherwise hex_park
    # collides with it instead of replacing it, leaving two same-named
    # columns and breaking the concat below.
    short = short.drop(columns=[pcols["hex_end"]]).rename(
        columns={pcols["hex_park"]: pcols["hex_end"]}
    )
    short = pd.DataFrame(
        short.drop(columns=[pcols["route_geom"], pcols["park_point"]])
    )
    short[pcols["routing_status"]] = "routed"

    # Bring all trips back together
    trips = pd.concat([orig, unrouted, short], axis=0)
    return trips


def get_optional_stop_trips(
    routes: dgpd.GeoDataFrame, parks: gpd.GeoDataFrame, params: dict
) -> dd.DataFrame:
    """Identify truck stops along each route and compute their distance from the trip origin.

    Fuses, into a single per-partition task, everything that used to be a
    chain of separate Dask operations: dropping unused columns, falling
    back to pre-routing ``trip_miles``/``trip_hrs`` for trips with no route
    geometry, buffering and spatially joining truck stops onto the route
    LineStrings, projecting each matched stop onto its route to get its
    distance from the trip origin, dropping the now-unneeded route
    geometry, and excluding optional stops too close to either trip
    endpoint. Returns a lazy ``dd.DataFrame`` -- nothing is computed here;
    the pipeline's only materialization point is the final catalog write.

    Args:
        routes: Routed trips Dask GeoDataFrame with route LineString geometry.
        parks: Truck-stop GeoDataFrame (output of ``format_stop_locations``).
        params: Pipeline parameters dict with keys:

            - ``columns`` (dict): sub-keys for column names including
              ``route_geom``, ``park_point``, ``park_id``, ``hex_end``,
              ``hex_park``, ``dist_along_miles``, ``routing_status``.
            - ``projected_crs`` (str | CRS): CRS used for buffering and
              distance projection.
            - ``park_buffer_miles`` (float): buffer radius around each truck
              stop (miles); also used to exclude stops adjacent to either
              trip endpoint.
            - ``drop_cols_initial`` (list[str]): columns to drop from the
              routes DataFrame before joining.

    Returns:
        A lazy ``dd.DataFrame`` combining original trips
        (``routing_status="original"`` or ``"unrouted"``) and optional
        truck-stop trips (``routing_status="routed"``), with a
        ``dist_along_miles`` column recording each record's distance from
        the trip origin.
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
    ``_build_partition_trips``; this first backfills distance/time/speed for
    ``routing_status="unrouted"`` rows (no GraphHopper route -- see
    ``_build_partition_trips``) from their pre-routing ``trip_miles``/
    ``trip_hrs``, then sorts by trip ID and distance from origin to
    establish sub-trip ordering, computes each sub-segment's distance as
    ``dist_along - dist_prev`` and its duration as ``(seg_miles /
    route_speed) x (obs_hours / route_hours)`` (proportional scaling
    preserves the observed total trip time), derives cumulative time shifts
    from the trip start time to produce new ``start_time``/``end_time`` for
    each sub-trip rounded to seconds, drops any split left with a duplicate
    ``end_time`` by that rounding, then renames/selects columns to match
    the original trips schema.

    ``trips`` arrives indexed by vehicle ID (set upstream by
    ``index_by_vehicle`` and carried through unchanged ever since, per
    ``_build_partition_trips``'s no-shuffle invariant) and stays indexed by
    it on the way out -- nothing downstream needs it back as a column, so
    there's no reason to round-trip it through one here. Sorting mirrors
    ``concat_optional_stops``: a stable sort on the remaining
    ``trip_id_cols`` plus ``dist_along_miles`` (the finest-grained
    tiebreaker) establishes sub-trip order, then a stable ``sort_index``
    groups rows back by vehicle ID while preserving that order for ties.
    ``groupby(trip_id_cols)`` below works unchanged either way -- pandas
    resolves names against index levels and columns together.
    """
    pcols = params["columns"]
    id_col, *rest_id_cols = params["trip_id_cols"]
    trip_id_cols = params["trip_id_cols"]

    trips = trips.sort_values(
        rest_id_cols + [pcols["dist_along_miles"]], ascending=True, kind="stable"
    ).sort_index(kind="stable")

    # "unrouted" rows (no GraphHopper route geometry -- see
    # _build_partition_trips) have no route-derived distance/time/speed;
    # fall back to the trip's pre-routing values so the segment math below
    # doesn't divide by zero or produce a NaN time_scaler, and so the
    # result reproduces the trip's originally observed start/end times
    # unchanged.
    is_unrouted = trips[pcols["routing_status"]] == "unrouted"
    trips.loc[is_unrouted, pcols["miles_route"]] = trips.loc[is_unrouted, pcols["miles_orig"]]
    trips.loc[is_unrouted, pcols["hours_route"]] = trips.loc[is_unrouted, pcols["hours_orig"]]
    trips.loc[is_unrouted, pcols["speed_route"]] = (
        trips.loc[is_unrouted, pcols["miles_route"]]
        / trips.loc[is_unrouted, pcols["hours_route"]]
    )
    trips.loc[is_unrouted, pcols["dist_along_miles"]] = trips.loc[is_unrouted, pcols["miles_route"]]

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

    # Stops close enough together to round to the same new_end which will cause
    # row explosions later when merging; trip_id_cols now names each split's own
    # end time (post-rename), so it doubles as the key for dropping repeats.
    key_arrays = [
        trips_out[col].to_numpy()
        if col in trips_out.columns
        else trips_out.index.get_level_values(col).to_numpy()
        for col in trip_id_cols
    ]
    is_dupe = pd.MultiIndex.from_arrays(key_arrays).duplicated(keep="first")
    trips_out = trips_out.loc[~is_dupe]

    # id_col (veh_id) stays the index rather than a column -- keep_cols_final
    # lists it because it names the original trips schema, but it's not a
    # selectable column here.
    keep_cols = [c for c in params["keep_cols_final"] if c != id_col]
    return trips_out.loc[:, keep_cols]


def describe_optional_stop_trips(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Compute split timestamps and distances for optional-stop sub-trips.

    Applies ``_describe_partition`` lazily to each partition of ``trips``
    (output of ``get_optional_stop_trips``). Safe to run per-partition
    because every trip's rows are guaranteed to live in one partition (see
    ``_build_partition_trips``'s docstring).

    The output stays indexed by vehicle ID, same as the input -- no shuffle
    happens here (see ``_build_partition_trips``'s no-shuffle invariant), so
    the existing divisions remain structurally valid. The catalog entry this
    feeds is checkpointed to disk with ``calculate_divisions: True`` on
    load, which recovers known divisions from the written partitions' index
    ranges for free -- no second ``index_by_vehicle``/``set_index`` pass is
    needed before ``concat_optional_stops``.

    Args:
        trips: Combined Dask DataFrame of original and optional-stop trips,
            indexed by vehicle ID (output of ``get_optional_stop_trips``).
        params: Pipeline parameters dict with keys:

            - ``columns`` (dict): sub-keys for ``dist_along_miles``,
              ``speed_route``, ``hours_orig``, ``hours_route``, ``start_time``,
              ``routing_status``, ``miles_route``, ``miles_orig``.
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
    trips_not_to_route: dd.DataFrame, trips_opt: dd.DataFrame, params: dict
) -> dd.DataFrame:
    """Merge optional-stop sub-trips with the trips that were never routed.

    ``trips_not_to_route`` (trips too short to route -- see
    ``select_trips_to_route``) and ``trips_opt`` (output of
    ``describe_optional_stop_trips``) are already disjoint by construction: a
    trip is either too short to route, or it went through routing and
    appears exactly once in ``trips_opt``, split or not. No anti-join/dedup
    is needed. Both inputs also arrive already indexed by vehicle ID, each
    with its own known divisions, so they can be combined via
    ``interleave_partitions=True`` -- a partition-wise merge on those
    divisions rather than a full hash shuffle.

    A final per-partition stable sort handles the remaining trip-ID
    column(s) within each vehicle's rows, since interleaving doesn't
    guarantee that ordering on its own.

    Args:
        trips_not_to_route: Trips too short to route, indexed by vehicle ID
            (output of ``select_trips_to_route``, which inherits the index
            set by ``index_by_vehicle`` on the full trips set upstream).
        trips_opt: Optional-stop sub-trips, indexed by vehicle ID (output of
            ``describe_optional_stop_trips``, checkpointed to disk with
            divisions recovered on load -- see that function's docstring).
        params: Pipeline parameters dict with keys:

            - ``trip_id_cols`` (list[str]): columns uniquely identifying a
              trip row; the leading column is the shared index; any
              remaining columns are sorted on per-partition below.
            - ``n_partitions`` (int): target partition count for the merged
              output. ``interleave_partitions=True`` builds output partitions
              from the union of both inputs' division boundaries, which can
              far exceed either input's own partition count; this rebalances
              it back down. Divisions stay known and monotonic going in, so
              this is a cheap merge of adjacent partitions, not a shuffle.

    Returns:
        A Dask DataFrame combining the never-routed trips and the described
        optional-stop trips, with one row per unique (trip_id_cols)
        combination, indexed by the leading ``trip_id_cols`` entry.
    """
    trips = dd.concat([trips_not_to_route, trips_opt], axis=0, interleave_partitions=True)
    trips = trips.repartition(npartitions=params["n_partitions"])

    _id_col, *rest_id_cols = params["trip_id_cols"]
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
