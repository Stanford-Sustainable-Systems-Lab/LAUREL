"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set
from megaPLuG.scenarios.io import (
    read_scenario_partition,
    write_scenario_partition,
)
from megaPLuG.utils.data import get_merge_params, merge_dataframes_node

from .nodes import (
    build_eval_columns,
    build_sampling_totals,
    build_slice_frame,
    filter_events,
    filter_slices_location,
    filter_slices_time,
    sample_vehicle_windows,
    slice_vehicle_windows,
    summarize_vehicle_window_quantiles,
    summarize_vehicles,
)


def create_pipeline(**kwargs) -> Pipeline:
    read_pipe = pipeline(
        [
            node(
                func=read_scenario_partition,
                inputs=["dwells_with_charging_partition", "params:results_partition"],
                outputs="dwells_with_charging_eval",
                name="collate_partitions_dwells_with_charging",
            ),
            node(
                func=load_dwell_set,
                inputs=["dwells_with_charging_eval", "params:load_dwell_set"],
                outputs="dwell_obj_eval",
                name="load_dwell_set_eval_impacts",
            ),
            node(
                func=read_scenario_partition,
                inputs=["vehicles_with_params_partition", "params:results_partition"],
                outputs="vehicles_with_params_eval",
                name="collate_partitions_vehicles_with_params",
            ),
            node(
                func=read_scenario_partition,
                inputs=["events_partition", "params:results_partition"],
                outputs="events_eval",
                name="collate_partitions_events",
            ),
            node(
                func=filter_events,
                inputs=["events_eval", "params:filter_events"],
                outputs="events_filtered",
                name="filter_events",
            ),
            node(
                func=build_slice_frame,
                inputs=["vehicles_evaluated", "params:build_slice_frame"],
                outputs="slice_frame",
                name="build_slice_frame",
            ),
        ],
        tags="scenario_run",
    )

    report_vehicles_pipe = pipeline(
        [
            node(
                func=summarize_vehicles,
                inputs=[
                    "dwell_obj_eval",
                    "vehicles_with_params_eval",
                    "params:summarize_vehicles",
                ],
                outputs="vehicles_evaluated",
                name="summarize_vehicles",
            ),
            node(
                func=write_scenario_partition,
                inputs=["vehicles_evaluated", "params:results_partition"],
                outputs="vehicles_evaluated_partition",
                name="write_scenario_partition_vehicles",
            ),
        ],
        tags=["report_vehicles", "scenario_run"],
    )

    report_profiles_scaled_pipe = pipeline(
        [
            node(
                func=build_eval_columns,
                inputs=["params:eval_columns", "params:group_columns"],
                outputs="eval_columns",
                name="build_eval_columns",
            ),
            node(
                func=get_merge_params,
                inputs=[
                    "params:assign_metadata_vehicle",
                    "vehicles_evaluated",
                    "params:stratify_columns",
                    "params:group_columns",
                ],
                outputs="merge_params_vehicles",
                name="get_merge_params_vehicles",
            ),
            node(
                func=get_merge_params,
                inputs=[
                    "params:assign_metadata_location",
                    "hex_region_corresp",
                    "params:stratify_columns",
                    "params:group_columns",
                ],
                outputs="merge_params_locations",
                name="get_merge_params_locations",
            ),
            node(
                func=merge_dataframes_node,
                inputs=[
                    "events_filtered",
                    "hex_region_corresp",
                    "merge_params_locations",
                ],
                outputs="events_w_regions",
                name="assign_metadata_location",
            ),
            node(
                func=merge_dataframes_node,
                inputs=[
                    "events_w_regions",
                    "vehicles_evaluated",
                    "merge_params_vehicles",
                ],
                outputs="events_w_metadata",
                name="assign_metadata_vehicles_to_events",
            ),
            node(
                func=merge_dataframes_node,
                inputs=[
                    "slice_frame",
                    "vehicles_evaluated",
                    "merge_params_vehicles",
                ],
                outputs="slice_frame_w_metadata",
                name="assign_metadata_vehicles_to_slice_frame",
            ),
            node(
                func=slice_vehicle_windows,
                inputs=[
                    "events_w_metadata",
                    "params:slice_events",
                    "eval_columns",
                ],
                outputs="slices",
                name="slice_vehicle_windows",
            ),
            node(
                func=filter_slices_location,
                inputs=["slices", "params:slice_events", "eval_columns"],
                outputs="slices_filtered_location",
                name="filter_slices_of_events_location",
            ),
            node(
                func=filter_slices_time,
                inputs=["slices", "params:slice_events", "eval_columns"],
                outputs="slices_filtered",
                name="filter_slices_of_events",
            ),
            node(
                func=filter_slices_time,
                inputs=[
                    "slice_frame_w_metadata",
                    "params:slice_events",
                    "eval_columns",
                ],
                outputs="slice_frame_filtered",
                name="filter_slices_of_slice_frame",
            ),
            node(
                func=build_sampling_totals,
                inputs=["vius_scaling", "params:build_sampling_totals"],
                outputs="sampling_totals",
                name="build_sampling_totals",
            ),
            node(
                func=sample_vehicle_windows,
                inputs=[
                    "slices_filtered",
                    "slice_frame_filtered",
                    "sampling_totals",
                    "params:slice_events",
                    "params:sample_slices",
                    "eval_columns",
                ],
                outputs="slices_sampled",
                name="sample_vehicle_windows",
            ),
            node(
                func=summarize_vehicle_window_quantiles,
                inputs=[
                    "slices_sampled",
                    "params:slice_events",
                    "params:summarize_slices",
                    "eval_columns",
                ],
                outputs="report_by_region_quantiles",
                name="summarize_vehicle_window_quantiles",
            ),
            node(
                func=write_scenario_partition,
                inputs=["report_by_region_quantiles", "params:results_partition"],
                outputs="report_by_region_quantiles_partition",
                name="write_scenario_partition_hexes_quants",
            ),
        ],
        tags="report_profiles",
    )

    profile_group_fixed_params = {
        "params:results_partition",
        "params:eval_columns",
        "params:slice_events",
        "params:sample_slices",
        "params:summarize_slices",
        "params:build_sampling_totals",
        "params:assign_metadata_vehicle",
        "params:assign_metadata_location",
        "params:stratify_columns",
    }
    profile_group_fixed_inputs = {
        "events_filtered",
        "slice_frame",
        "hex_region_corresp",
        "vehicles_evaluated",
        "vius_scaling",
    }

    report_profiles_pipes = [
        pipeline(
            report_profiles_scaled_pipe,
            namespace="substation",
            parameters=profile_group_fixed_params,
            inputs=profile_group_fixed_inputs,
            tags="scenario_run",
        ),
        pipeline(
            report_profiles_scaled_pipe,
            namespace="jurisdiction",
            parameters=profile_group_fixed_params,
            inputs=profile_group_fixed_inputs,
            tags="scenario_run",
        ),
    ]

    return read_pipe + report_vehicles_pipe + sum(report_profiles_pipes)
