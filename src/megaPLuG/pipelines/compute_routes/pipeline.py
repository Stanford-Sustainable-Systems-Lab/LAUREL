"""
This is a boilerplate pipeline 'compute_routes'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_dwell_id,
    filter_routable_dwells_before_geoms,
    filter_routable_trips,
    get_routes_node,
    get_trip_origs_and_dests,
    import_graph,
)


def create_pipeline(**kwargs) -> Pipeline:
    import_pipe = pipeline(
        [
            node(
                func=import_graph,
                inputs="params:graphhopper",
                outputs=None,
                name="import_graph",
            ),
        ],
        tags="import",
    )

    route_pipe = pipeline(
        [
            node(
                func=build_dwell_id,
                inputs=["dwells_with_locations", "params:build_dwell_id"],
                outputs="dwells_with_ids",
                name="build_dwell_id",
            ),
            node(
                func=filter_routable_dwells_before_geoms,
                inputs=["dwells_with_ids", "params:filter_routable_dwells"],
                outputs="dwells_prefiltered",
                name="filter_routable_dwells_before_geoms",
            ),
            node(
                func=get_trip_origs_and_dests,
                inputs=["dwells_prefiltered", "params:get_trip_origs_and_dests"],
                outputs="dwells_orig_dest",
                name="get_trip_origs_and_dests",
            ),
            node(
                func=filter_routable_trips,
                inputs=["dwells_orig_dest", "params:filter_routable_trips"],
                outputs="dwells_orig_dest_filtered",
                name="filter_routable_trips",
            ),
            node(
                func=get_routes_node,
                inputs=[
                    "dwells_orig_dest_filtered",
                    "params:get_routes",
                    "params:graphhopper",
                ],
                outputs="dwells_with_routes",
                name="get_routes",
            ),
        ],
        tags="route",
    )

    return import_pipe + route_pipe
