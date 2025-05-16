"""
This is a boilerplate pipeline 'compute_routes'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import import_graph, get_routes


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
        tags="import"
    )

    route_pipe = pipeline(
        [
            node(
                func=get_routes,
                inputs=["params:get_routes", "params:graphhopper"],
                outputs=None,
                name="get_routes",
            ),
        ],
        tags="route"
    )

    return import_pipe + route_pipe
