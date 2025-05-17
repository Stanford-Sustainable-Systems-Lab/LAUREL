"""
This is a boilerplate pipeline 'compute_routes'
generated using Kedro 0.19.3
"""

from megaPLuG.models.routing.router import GraphhopperContainerRouter
from routingpy import Graphhopper

import logging

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
    )
    server.import_graph(input_file=server_params["map_path"])
    logger.info("Import completed")


def get_routes(route_params: dict, server_params: dict) -> None:
    """Get routes using Graphhopper."""
    resource = server_params["resources"]["server"]
    with GraphhopperContainerRouter(
        image=server_params["image"],
        graph_dir=server_params["graph_dir"],
        config_path=server_params["config_path"],
        mem_max_gb=resource["mem_max_gb"],
        mem_start_gb=resource["mem_start_gb"],
    ) as server:
        router = Graphhopper(base_url=server.base_url)
        route = router.directions(locations=coords, profile=route_params["profile"])
        logger.info(f"Route distance: {route.distance} meters")
