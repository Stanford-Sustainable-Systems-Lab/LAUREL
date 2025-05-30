"""
This is a boilerplate pipeline 'compute_routes'
generated using Kedro 0.19.3
"""

import logging

import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
from routingpy import Graphhopper

from megaPLuG.models.routing.router import (
    DIST_COL,
    ROUTE_COL,
    TIME_COL,
    get_routes,
)
from megaPLuG.models.routing.server import GraphhopperContainerRouter
from megaPLuG.utils.geo import METERS_PER_MILE
from megaPLuG.utils.h3 import add_geometries
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


def build_dwell_id(dwells: pd.DataFrame, params: dict) -> pd.DataFrame:
    dwells[params["col_name"]] = pd.RangeIndex(stop=dwells.shape[0])
    return dwells


def filter_routable_dwells_before_geoms(
    dwells: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Filter down routable dwells without using geometries (to speed geometry creation)."""
    if params["apply_filter"]:
        vehs_sel = dwells.index.unique()[: params["n_vehs"]]
        dwells_filt = dwells.loc[vehs_sel]
        return dwells_filt
    else:
        return dwells


def get_trip_origs_and_dests(dwells: pd.DataFrame, params: dict) -> gpd.GeoDataFrame:
    """Get origins and destinations for each dwell, and format for routing."""
    pcols = params["columns"]
    trips = add_geometries(data=dwells, hex_col=pcols["hex"])
    trips = trips.rename_geometry(pcols["dest"])
    trips[pcols["orig"]] = trips.groupby(pcols["veh"])[pcols["dest"]].shift(1)
    return trips


def filter_routable_trips(dwells: pd.DataFrame, params: dict) -> gpd.GeoDataFrame:
    """Filter down routable dwells using the geometries."""
    dwells_filt = dwells.loc[dwells[params["dist_col"]] >= params["min_dist_miles"]]
    return dwells_filt


def partition_trips(trips: gpd.GeoDataFrame, params: dict) -> dgpd.GeoDataFrame:
    """Partition the trips to save to disk, in preparation for routing."""
    parts = dgpd.from_geopandas(trips, chunksize=params["n_trips_per_partition"])
    return parts


def get_routes_node(
    trips: gpd.GeoDataFrame,
    server: GraphhopperContainerRouter,
    params: dict,
) -> gpd.GeoDataFrame:
    """Compute routes for each dwell and then format results.

    Also returns the server so that it can be stopped afterward. This is essential for
    Kedro to manage the server nodes in order.
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
        server_url=server.base_url,
        profile=params["profile"],
    )
    # routed = routed.compute()
    logger.info("Finished routing")

    logger.info("Interpreting routes")
    tcols = params["output_trip_cols"]
    float_type = pd.Float64Dtype()
    routed[tcols["dist"]] = (routed[DIST_COL] / METERS_PER_MILE).astype(float_type)
    routed[tcols["dur"]] = (routed[TIME_COL] / SECS_PER_HOUR).astype(float_type)
    routed[tcols["speed"]] = routed[tcols["dist"]] / routed[tcols["dur"]]
    routed = routed.drop(columns=[DIST_COL, TIME_COL])
    routed = routed.set_geometry(ROUTE_COL)
    return routed
