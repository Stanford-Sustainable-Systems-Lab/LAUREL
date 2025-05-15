"""
This is a boilerplate pipeline 'compute_routes'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import manage_container


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=manage_container,
                inputs="params:manage_container",
                outputs=None,
                name="manage_container",
            ),
        ],
    )
    return pipe
