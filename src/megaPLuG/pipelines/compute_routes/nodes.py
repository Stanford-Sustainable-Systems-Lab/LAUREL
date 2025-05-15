"""
This is a boilerplate pipeline 'compute_routes'
generated using Kedro 0.19.3
"""

from megaPLuG.models.routing.router import GraphhopperContainerRouter


def manage_container(server_params: dict) -> None:
    """Learn to manage containers."""
    server = GraphhopperContainerRouter(
        image=server_params["image"], graph_dir=server_params["graph_dir"]
    )
    server.import_graph(url=server_params["graph_url"])
    print("hello world")
