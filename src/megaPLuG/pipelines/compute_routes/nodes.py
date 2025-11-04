"""
This is a boilerplate pipeline 'compute_routes'
generated using Kedro 0.19.3
"""

import logging

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.diagnostics.progress import ProgressBar
from routingpy import Graphhopper

from megaPLuG.models.routing.router import (
    DIST_COL,
    ROUTE_COL,
    TIME_COL,
    get_routes,
)
from megaPLuG.models.routing.server import GraphhopperContainerRouter
from megaPLuG.utils.geo import METERS_PER_MILE
from megaPLuG.utils.h3 import add_geometries, cells_to_points
from megaPLuG.utils.time import SECS_PER_HOUR

logger = logging.getLogger(__name__)


def import_graph(server_params: dict) -> None:
    """Import a Graphhopper graph."""
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
    """Get routes using Graphhopper."""
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
    """Filter down routable trips using the geometries."""
    trips = trips.drop(columns=params["drop_cols"])
    long_enough_trip = trips[params["dist_col"]] >= params["min_dist_miles"]
    trips = trips.loc[long_enough_trip]
    if params["debug_subsample"]["active"]:
        trips = trips.sample(frac=params["debug_subsample"]["frac"])
    return trips


def get_trip_orig_dest_points(trips: dd.DataFrame, params: dict) -> dgpd.GeoDataFrame:
    """Get origin and destination points for each trip."""
    trips = dgpd.from_dask_dataframe(trips, geometry=None)
    for tgt, src in params["hex_geo_cols"].items():
        trips[tgt] = trips[src].map_partitions(cells_to_points, meta=gpd.GeoSeries())
    trips = trips.set_geometry(params["output_geom_col"])
    return trips


def partition_trips(trips: dgpd.GeoDataFrame, params: dict) -> dgpd.GeoDataFrame:
    """Re-partition the trips to save to disk, in preparation for routing."""
    parts = trips.repartition(npartitions=params["n_partitions"])
    return parts


def get_routes_node(
    trips: dgpd.GeoDataFrame,
    server: GraphhopperContainerRouter,
    params: dict,
) -> dgpd.GeoDataFrame:
    """Compute routes for each dwell and then format results."""
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
    """Concatenate together the locations for optional stops from several sources."""
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
    """Compute the optional trip stops and their distances along the routes."""
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
    """Describe the new timings and distances for optional and original trips."""
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
    """Concatenate new optional trips onto original trips."""
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
    return trips
