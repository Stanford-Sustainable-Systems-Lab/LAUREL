"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calc_derived_trip_cols,
    create_dwells,
    format_trips_columns,
    get_vius_from_url,
    strip_vehicle_attrs,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=format_trips_columns,
                inputs=["navistar", "params:format_columns"],
                outputs="trips_formatted",
                name="format_trips_columns",
            ),
            node(
                func=strip_vehicle_attrs,
                inputs=["trips_formatted", "params:strip_vehicle_attrs"],
                outputs=["trips_stripped", "vehicles_raw"],
                name="strip_vehicle_attrs",
            ),
            node(
                func=calc_derived_trip_cols,
                inputs=["trips_stripped", "params:trip_derived_cols"],
                outputs="trips_derived",
                name="calc_derived_trip_cols",
            ),
            node(
                func=create_dwells,
                inputs=["trips_derived", "params:create_dwells"],
                outputs="dwells",
                name="create_dwells",
            ),
            node(
                func=get_vius_from_url,
                inputs=[
                    "vius_home_base_state_raw",
                    "params:get_vius_by_home_base_state",
                ],
                outputs="vius_home_base_state",
                name="get_vius_by_home_base_state",
                tags="downloads",
            ),
            node(
                func=get_vius_from_url,
                inputs=[
                    "vius_weight_class_raw",
                    "params:get_vius_by_weight_class",
                ],
                outputs="vius_weight_class",
                name="get_vius_by_weight_class",
                tags="downloads",
            ),
        ],
    )
    return pipe
