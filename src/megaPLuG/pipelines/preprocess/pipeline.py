"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_vius_scaling_totals,
    calc_derived_trip_cols,
    concat_optional_stops,
    create_dwells,
    describe_optional_stop_trips,
    format_trips_columns,
    get_optional_stop_trips,
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
                    "trips_routed",  # Use the `compute_routes` pipeline to get this
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
        tags="create_dwells_optional_stops",
    )

    dwell_pipe = pipeline(
        [
            # If you want optional stops, then use "trips_with_optional" as the input.
            # Otherwise, use "trips_formatted". Also, if you want optional stops, run
            # the `optional_stops` and this `create_dwells` pipeline together, using the
            # "create_dwells_optional_stops" tag for convenience.
            node(
                func=create_dwells,
                inputs=["trips_with_optional", "params:create_dwells"],
                outputs="dwells",
                name="create_dwells",
            ),
        ],
        tags=["create_dwells", "create_dwells_optional_stops"],
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

    return format_pipe + veh_pipe + opt_stops_pipe + dwell_pipe + scale_pipe
