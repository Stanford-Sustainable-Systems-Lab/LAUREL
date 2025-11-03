from kedro.pipeline import Node, Pipeline  # noqa

from megaPLuG.models.dwell_sets import save_dwell_set

from .nodes import (
    coalesce_interrupted_dwells,
    create_dwells,
    calc_derived_trip_cols,
    format_trips_columns,
    calc_rolling_dwell_ratios,
    map_location_groups,
)

from megaPLuG.utils.distributed import start_dask_node


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

    dwell_pipe = Pipeline(
        [
            Node(
                func=start_dask_node,
                inputs=["params:dask_describe_dwells"],
                outputs=["dask_cluster_dwells", "dask_client_dwells"],
                name="start_dask_node_describe_dwells",
            ),
            # If you want optional stops, then use "trips_with_optional" as the input.
            # Otherwise, use "trips_formatted". Also, if you want optional stops, run
            # the `optional_stops` and this `create_dwells` pipeline together, using the
            # "create_dwells_optional_stops" tag for convenience.
            Node(
                func=create_dwells,
                inputs=[
                    "trips_formatted",
                    "params:create_dwells",
                    "dask_client_dwells",
                ],
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
                func=calc_rolling_dwell_ratios,
                inputs=[
                    "dwell_obj_coalesced",
                    "params:rolling_dwell_ratios",
                ],
                outputs="dwell_obj_roll",
                name="calc_rolling_dwell_ratios",
            ),
            Node(
                func=map_location_groups,
                inputs=[
                    "dwell_obj_roll",
                    "hex_cluster_corresp",
                    "params:map_location_groups",
                ],
                outputs="dwell_obj_loc_groups",
                name="map_location_groups",
            ),
            Node(
                func=save_dwell_set,
                inputs="dwell_obj_loc_groups",
                outputs="dwells_with_locations_dask",
                name="save_dwell_set_preprocess",
            ),
        ],
        tags=["create_dwells", "create_dwells_optional_stops"],
    )

    return format_pipe + dwell_pipe
