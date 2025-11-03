"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    build_vius_scaling_totals,
    calc_derived_trip_cols,
    format_trips_columns,
    strip_vehicle_attrs,
)


def create_pipeline(**kwargs) -> Pipeline:
    format_pipe = Pipeline(
        [
            Node(
                func=format_trips_columns,
                inputs=["navistar", "params:format_columns"],
                outputs="trips_formatted_no_derived",
                name="format_trips_columns",
            ),
            Node(
                func=calc_derived_trip_cols,
                inputs=["trips_formatted_no_derived", "params:trip_derived_cols"],
                outputs="trips_formatted",
                name="calc_derived_trip_cols",
            ),
        ],
        tags="format_trips",
    )

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

    return format_pipe + veh_pipe + scale_pipe
