"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    build_vius_scaling_totals,
    strip_vehicle_attrs,
)


def create_pipeline(**kwargs) -> Pipeline:
    veh_pipe = Pipeline(
        [
            Node(
                func=strip_vehicle_attrs,
                inputs=["trips_formatted", "params:strip_vehicle_attrs"],
                outputs="vehicles_raw",
                name="strip_vehicle_attrs",
            ),
        ],
        tags="strip_vehicles",
    )

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

    return veh_pipe + scale_pipe
