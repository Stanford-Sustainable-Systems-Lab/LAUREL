"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.routing.nodes import (
    start_routing_server_node,
    stop_routing_server_node,
)
from megaPLuG.utils.distributed import start_dask_node, stop_dask_node

from .nodes import (
    build_vius_scaling_totals,
    calc_derived_trip_cols,
    create_dwells,
    filter_routable_trips,
    format_trips_columns,
    get_routes_node,
    get_trip_orig_dest_points,
    partition_trips,
    strip_vehicle_attrs,
)


def create_pipeline(**kwargs) -> Pipeline:
    format_pipe = pipeline(
        [
            node(
                func=format_trips_columns,
                inputs=["navistar", "params:format_columns"],
                outputs="trips_formatted",
                name="format_trips_columns",
            ),
        ],
        tags="format_trips",
    )

    veh_pipe = pipeline(
        [
            node(
                func=strip_vehicle_attrs,
                inputs=["trips_formatted", "params:strip_vehicle_attrs"],
                outputs="vehicles_raw",
                name="strip_vehicle_attrs",
            ),
        ],
        tags="strip_vehicles",
    )

    pre_route_pipe = pipeline(
        [
            node(
                func=filter_routable_trips,
                inputs=["trips_formatted", "params:filter_routable_trips"],
                outputs="trips_routable_filtered",
                name="filter_routable_trips_preprocess",
            ),
            node(
                func=get_trip_orig_dest_points,
                inputs=["trips_routable_filtered", "params:get_trip_orig_dest_points"],
                outputs="trips_to_route_big_partitions",
                name="get_trip_orig_dest_points",
            ),
            node(
                func=partition_trips,
                inputs=["trips_to_route_big_partitions", "params:partition_trips"],
                outputs="trips_to_route",
                name="partition_trips_preprocess",
            ),
        ],
        tags="pre_routing",
    )

    route_pipe = pipeline(
        [
            node(
                func=start_dask_node,
                inputs="params:dask_routing",
                outputs=["dask_cluster_routing", "dask_client_routing"],
                name="start_dask_routing",
            ),
            node(
                func=start_routing_server_node,
                inputs=[
                    "params:graphhopper",
                ],
                outputs="routing_server",
                name="start_routing_server",
            ),
            node(
                func=get_routes_node,
                inputs=[
                    "trips_to_route",
                    "routing_server",
                    "params:get_routes",
                ],
                outputs="trips_routed",
                name="get_routes_preprocess",
            ),
            node(
                func=stop_routing_server_node,
                inputs=["routing_server", "trips_routed"],
                outputs=None,
                name="stop_routing_server_preprocess",
            ),
            node(
                func=stop_dask_node,
                inputs=["dask_cluster_routing", "dask_client_routing", "trips_routed"],
                outputs=None,
                name="stop_dask_routing",
            ),
        ],
        tags="routing",
    )

    dwell_pipe = pipeline(
        [
            node(
                func=calc_derived_trip_cols,
                inputs=["trips_formatted", "params:trip_derived_cols"],
                outputs="trips_derived",
                name="calc_derived_trip_cols",
            ),
            node(
                func=create_dwells,
                inputs=["trips_derived", "params:create_dwells"],
                outputs="dwells",
                name="create_dwells",
            ),
        ],
        tags="create_dwells",
    )

    scale_pipe = pipeline(
        [
            node(
                func=build_vius_scaling_totals,
                inputs=[
                    "vius_public_use",
                    "params:build_vius_scaling_totals",
                ],
                outputs="vius_scaling",
                name="build_vius_scaling_totals",
            ),
        ],
        tags="vius_scaling",
    )

    return (
        format_pipe + veh_pipe + dwell_pipe + scale_pipe + pre_route_pipe + route_pipe
    )
