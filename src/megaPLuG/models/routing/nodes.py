from megaplug.models.routing.server import GraphhopperContainerRouter
from megaplug.utils.params import import_from_config


def start_routing_server_node(params: dict) -> GraphhopperContainerRouter:
    """Start the routing server and return the server object."""
    resource = params["resources"]["server"]
    server = GraphhopperContainerRouter(
        image=params["image"],
        runner_class=import_from_config(
            params["container_class"]
        ),  # TODO: Something about this doesn't work with import_from_config
        graph_dir=params["graph_dir"],
        config_path=params["config_path"],
        mem_max_gb=resource["mem_max_gb"],
        mem_start_gb=resource["mem_start_gb"],
        startup_delay=resource["startup_delay_secs"],
    )
    server = server.__enter__()
    return server


def stop_routing_server_node(
    server: GraphhopperContainerRouter, result: object
) -> None:
    """Stop the routing server.

    result is used to ensure that this node runs last, after all desired results have
    been computed. Pass the final dataset which requires the routing server to this node.
    """
    server.__exit__(None, None, None)
