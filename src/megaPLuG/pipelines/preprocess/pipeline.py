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
    concat_optional_stops,
    create_dwells,
    describe_optional_stop_trips,
    filter_routable_trips,
    format_trips_columns,
    get_optional_stop_trips,
    get_routes_node,
    get_trip_orig_dest_points,
    partition_trips,
    prepare_stop_locations,
    strip_vehicle_attrs,
)


def create_pipeline(**kwargs) -> Pipeline:
    format_pipe = pipeline(
        [
            node(
                func=format_trips_columns,
                inputs=["navistar", "params:format_columns"],
                outputs="trips_formatted_no_derived",
                name="format_trips_columns",
            ),
            node(
                func=calc_derived_trip_cols,
                inputs=["trips_formatted_no_derived", "params:trip_derived_cols"],
                outputs="trips_formatted",
                name="calc_derived_trip_cols",
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

    opt_stops_pipe = pipeline(
        [
            node(
                func=prepare_stop_locations,
                inputs=["parking", "params:prepare_stop_locations"],
                outputs="parking_formatted",
                name="prepare_stop_locations",
            ),
            node(
                func=get_optional_stop_trips,
                inputs=[
                    "trips_routed",
                    "parking_formatted",
                    "params:get_optional_stop_trips",
                ],
                outputs="optional_stop_trips_raw",
                name="get_optional_stop_trips",
            ),
            node(
                func=describe_optional_stop_trips,
                inputs=[
                    "optional_stop_trips_raw",
                    "params:describe_optional_stop_trips",
                ],
                outputs="optional_stop_trips",
                name="describe_optional_stop_trips",
            ),
            node(
                func=concat_optional_stops,
                inputs=[
                    "trips_formatted",
                    "optional_stop_trips",
                    "params:concat_optional_stops",
                ],
                outputs="trips_with_optional",
                name="concat_optional_stops",
            ),
        ],
        tags="optional_stops",
    )

    dwell_pipe = pipeline(
        [
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
        format_pipe
        + veh_pipe
        + dwell_pipe
        + scale_pipe
        + pre_route_pipe
        + route_pipe
        + opt_stops_pipe
    )
