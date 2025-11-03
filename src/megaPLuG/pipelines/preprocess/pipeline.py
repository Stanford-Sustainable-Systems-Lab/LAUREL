"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    build_vius_scaling_totals,
)


def create_pipeline(**kwargs) -> Pipeline:
    scale_pipe = Pipeline(
        [
            Node(
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

    return scale_pipe
