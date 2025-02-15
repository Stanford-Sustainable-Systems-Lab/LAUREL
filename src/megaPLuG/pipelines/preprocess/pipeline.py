"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_vius_scaling_totals,
    calc_derived_trip_cols,
    create_dwells,
    describe_substation_usage,
    format_substation_boundaries,
    format_substation_profiles,
    format_trips_columns,
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
                func=build_vius_scaling_totals,
                inputs=[
                    "vius_public_use",
                    "params:build_vius_scaling_totals",
                ],
                outputs="vius_scaling",
                name="build_vius_scaling_totals",
            ),
        ],
    )

    capacity_pipe = pipeline(
        [
            node(
                func=format_substation_boundaries,
                inputs=["substation_boundaries", "params:format_substation_capacities"],
                outputs="substation_boundaries_formatted",
                name="format_substation_boundaries",
            ),
            node(
                func=format_substation_profiles,
                inputs=["substation_profiles", "params:format_substation_profiles"],
                outputs="substation_profiles_formatted",
                name="format_substation_profiles",
            ),
            node(
                func=describe_substation_usage,
                inputs=[
                    "substation_profiles_formatted",
                    "substation_boundaries_formatted",
                    "params:describe_substation_usage",
                ],
                outputs="substation.usage",
                name="describe_substation_usage",
            ),
        ],
        tags="substation_usage",
    )

    return pipe + capacity_pipe
