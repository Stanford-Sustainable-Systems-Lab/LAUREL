"""
This is a boilerplate pipeline 'compute_routes'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Node, Pipeline

from megaPLuG.models.routing.nodes import (
    start_routing_server_node,
    stop_routing_server_node,
)
from megaPLuG.utils.distributed import start_dask_node, stop_dask_node

from .nodes import (
    filter_routable_trips,
    get_routes_node,
    get_trip_orig_dest_points,
    import_graph,
    partition_trips,
)


def create_pipeline(**kwargs) -> Pipeline:
    import_pipe = Pipeline(
        [
            Node(
                func=import_graph,
                inputs="params:graphhopper",
                outputs=None,
                name="import_graph",
            ),
        ],
        tags="import",
    )

    pre_route_pipe = Pipeline(
        [
            Node(
                func=filter_routable_trips,
                inputs=["trips_formatted", "params:filter_routable_trips"],
                outputs="trips_routable_filtered",
                name="filter_routable_trips",
            ),
            Node(
                func=get_trip_orig_dest_points,
                inputs=["trips_routable_filtered", "params:get_trip_orig_dest_points"],
                outputs="trips_to_route_big_partitions",
                name="get_trip_orig_dest_points",
            ),
            Node(
                func=partition_trips,
                inputs=["trips_to_route_big_partitions", "params:partition_trips"],
                outputs="trips_to_route",
                name="partition_trips",
            ),
        ],
        tags="pre_routing",
    )

    route_pipe = Pipeline(
        [
            Node(
                func=start_dask_node,
                inputs="params:dask_routing",
                outputs=["dask_cluster_routing", "dask_client_routing"],
                name="start_dask_routing",
            ),
            Node(
                func=start_routing_server_node,
                inputs=[
                    "params:graphhopper",
                ],
                outputs="routing_server",
                name="start_routing_server",
            ),
            Node(
                func=get_routes_node,
                inputs=[
                    "trips_to_route",
                    "routing_server",
                    "params:get_routes",
                ],
                outputs="trips_routed",
                name="get_routes",
            ),
            Node(
                func=stop_routing_server_node,
                inputs=["routing_server", "trips_routed"],
                outputs=None,
                name="stop_routing_server",
            ),
            Node(
                func=stop_dask_node,
                inputs=["dask_cluster_routing", "dask_client_routing", "trips_routed"],
                outputs=None,
                name="stop_dask_routing",
            ),
        ],
        tags="routing",
    )

    return import_pipe + pre_route_pipe + route_pipe
