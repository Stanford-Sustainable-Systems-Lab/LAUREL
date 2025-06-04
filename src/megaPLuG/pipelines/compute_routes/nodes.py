"""
This is a boilerplate pipeline 'compute_routes'
generated using Kedro 0.19.3
"""

import logging

import dask.dataframe as dd
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
from megaPLuG.utils.h3 import cells_to_points
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
    float_type = pd.Float64Dtype()
    routed[tcols["dist"]] = (routed[DIST_COL] / METERS_PER_MILE).astype(float_type)
    routed[tcols["dur"]] = (routed[TIME_COL] / SECS_PER_HOUR).astype(float_type)
    routed[tcols["speed"]] = routed[tcols["dist"]] / routed[tcols["dur"]]
    routed = routed.drop(columns=[DIST_COL, TIME_COL])
    routed = routed.set_geometry(ROUTE_COL)
    return routed
