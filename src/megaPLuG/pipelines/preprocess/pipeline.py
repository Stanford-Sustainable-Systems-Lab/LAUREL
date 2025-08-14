"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Node, Pipeline

from megaPLuG.models.dwell_sets import save_dwell_set

from .nodes import (
    build_vius_scaling_totals,
    calc_derived_trip_cols,
    coalesce_interrupted_dwells,
    concat_optional_stops,
    concat_stop_locations,
    create_dwells,
    describe_optional_stop_trips,
    format_trips_columns,
    get_optional_stop_trips,
    prepare_stop_locations_private,
    prepare_stop_locations_public,
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

    opt_stops_pipe = Pipeline(
        [
            Node(
                func=prepare_stop_locations_public,
                inputs=["parking_public", "params:prepare_stop_locations_public"],
                outputs="parking_formatted_public",
                name="prepare_stop_locations_public",
            ),
            Node(
                func=prepare_stop_locations_private,
                inputs=["parking_private", "params:prepare_stop_locations_private"],
                outputs="parking_formatted_private",
                name="prepare_stop_locations_private",
            ),
            Node(
                func=concat_stop_locations,
                inputs=[
                    "parking_formatted_public",
                    "parking_formatted_private",
                    "params:concat_stop_locations",
                ],
                outputs="parking_formatted",
                name="concat_stop_locations",
            ),
            Node(
                func=get_optional_stop_trips,
                inputs=[
                    "trips_routed",  # Use the `compute_routes` pipeline to get this
                    "parking_formatted",
                    "params:get_optional_stop_trips",
                ],
                outputs="optional_stop_trips_raw",
                name="get_optional_stop_trips",
            ),
            Node(
                func=describe_optional_stop_trips,
                inputs=[
                    "optional_stop_trips_raw",
                    "params:describe_optional_stop_trips",
                ],
                outputs="optional_stop_trips",
                name="describe_optional_stop_trips",
            ),
            Node(
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

    dwell_pipe = Pipeline(
        [
            # If you want optional stops, then use "trips_with_optional" as the input.
            # Otherwise, use "trips_formatted". Also, if you want optional stops, run
            # the `optional_stops` and this `create_dwells` pipeline together, using the
            # "create_dwells_optional_stops" tag for convenience.
            Node(
                func=create_dwells,
                inputs=["trips_with_optional", "params:create_dwells"],
                outputs="dwell_obj_preprocess",
                name="create_dwells",
            ),
            Node(
                func=coalesce_interrupted_dwells,
                inputs=["dwell_obj_preprocess", "params:coalesce_interrupted_dwells"],
                outputs="dwell_obj_coalesced",
                name="coalesce_interrupted_dwells",
            ),
            Node(
                func=save_dwell_set,
                inputs="dwell_obj_coalesced",
                outputs="dwells",
                name="save_dwell_set_preprocess",
            ),
        ],
        tags=["create_dwells", "create_dwells_optional_stops"],
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

    return format_pipe + veh_pipe + opt_stops_pipe + dwell_pipe + scale_pipe
