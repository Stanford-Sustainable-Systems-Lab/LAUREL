"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Node, Pipeline

from megaPLuG.models.dwell_sets import load_dwell_set
from megaPLuG.scenarios.io import (
    read_scenario_partition,
    write_scenario_partition,
)
from megaPLuG.utils.data import (
    categorize_columns,
    get_merge_params,
    merge_dataframes_node,
)
from megaPLuG.utils.distributed import start_dask_node

from .nodes import (
    build_eval_columns,
    build_sampling_totals,
    build_slice_frame,
    filter_events,
    filter_slices_time,
    localize_time_from_hexes,
    sample_vehicle_windows,
    slice_vehicle_windows,
    summarize_vehicle_window_quantiles,
    summarize_vehicles,
)


def create_pipeline(**kwargs) -> Pipeline:
    read_pipe = Pipeline(
        [
            Node(
                func=read_scenario_partition,
                inputs=["dwells_with_charging_partition", "params:results_partition"],
                outputs="dwells_with_charging_eval",
                name="collate_partitions_dwells_with_charging",
            ),
            Node(
                func=load_dwell_set,
                inputs=["dwells_with_charging_eval", "params:load_dwell_set"],
                outputs="dwell_obj_eval",
                name="load_dwell_set_eval_impacts",
            ),
            Node(
                func=read_scenario_partition,
                inputs=["vehicles_with_params_partition", "params:results_partition"],
                outputs="vehicles_with_params_eval",
                name="collate_partitions_vehicles_with_params",
            ),
            Node(
                func=categorize_columns,
                inputs="vehicles_with_params_eval",
                outputs="vehicles_with_params_eval_categorized",
                name="categorize_vehicles_with_params_eval",
            ),
            Node(
                func=read_scenario_partition,
                inputs=["events_partition", "params:results_partition"],
                outputs="events_eval",
                name="collate_partitions_events",
            ),
            Node(
                func=read_scenario_partition,
                inputs=["hex_region_corresp_partition", "params:geo_partition_eval"],
                outputs="hex_region_corresp",
                name="collate_partitions_hexes",
            ),
            Node(
                func=categorize_columns,
                inputs="hex_region_corresp",
                outputs="hex_region_corresp_categorized",
                name="categorize_hex_region_corresp",
            ),
        ],
        tags="scenario_run",
    )

    report_vehicles_pipe = Pipeline(
        [
            Node(
                func=summarize_vehicles,
                inputs=[
                    "dwell_obj_eval",
                    "vehicles_with_params_eval_categorized",
                    "params:summarize_vehicles",
                ],
                outputs="vehicles_evaluated",
                name="summarize_vehicles",
            ),
            Node(
                func=write_scenario_partition,
                inputs=["vehicles_evaluated", "params:results_partition"],
                outputs="vehicles_evaluated_partition",
                name="write_scenario_partition_vehicles",
            ),
        ],
        tags=["report_vehicles", "scenario_run"],
    )

    report_profiles_scaled_prep_pipe = Pipeline(
        [
            Node(
                func=get_merge_params,
                inputs=[
                    "params:assign_metadata_location",
                    "hex_region_corresp_categorized",
                    "params:stratify_columns",
                    "params:substation.group_columns",
                    "params:state_op_dist.group_columns",
                ],
                outputs="merge_params_locations",
                name="get_merge_params_locations",
            ),
            Node(
                func=merge_dataframes_node,
                inputs=[
                    "events_eval",
                    "hex_region_corresp_categorized",
                    "merge_params_locations",
                ],
                outputs="events_w_regions",
                name="assign_metadata_location",
            ),
            Node(
                func=get_merge_params,
                inputs=[
                    "params:assign_metadata_vehicle",
                    "vehicles_evaluated",
                    "params:stratify_columns",
                    "params:substation.group_columns",
                    "params:state_op_dist.group_columns",
                ],
                outputs="merge_params_vehicles",
                name="get_merge_params_vehicles",
            ),
            Node(
                func=merge_dataframes_node,
                inputs=[
                    "events_w_regions",
                    "vehicles_evaluated",
                    "merge_params_vehicles",
                ],
                outputs="events_w_metadata",
                name="assign_metadata_vehicles_to_events",
            ),
            Node(
                func=filter_events,  # TODO: Add location filtering to this
                inputs=["events_w_metadata", "params:filter_events"],
                outputs="events_filtered",
                name="filter_events",
            ),
            Node(
                func=localize_time_from_hexes,
                inputs=[
                    "events_filtered",
                    "params:localize_time_from_hexes",
                    "params:eval_columns",
                ],
                outputs="events_w_local_time",
                name="localize_time_from_hexes",
            ),
            Node(
                func=slice_vehicle_windows,
                inputs=[
                    "events_w_local_time",
                    "params:slice_events",
                    "params:eval_columns",
                ],
                outputs="slices",
                name="slice_vehicle_windows",
            ),
            Node(
                func=filter_slices_time,
                inputs=["slices", "params:slice_events", "params:eval_columns"],
                outputs="slices_filtered",
                name="filter_slices_of_events",
            ),
            Node(
                func=build_slice_frame,
                inputs=["vehicles_evaluated", "params:build_slice_frame"],
                outputs="slice_frame",
                name="build_slice_frame",
            ),
            Node(
                func=merge_dataframes_node,
                inputs=[
                    "slice_frame",
                    "vehicles_evaluated",
                    "merge_params_vehicles",
                ],
                outputs="slice_frame_w_metadata",
                name="assign_metadata_vehicles_to_slice_frame",
            ),
            Node(
                func=filter_slices_time,
                inputs=[
                    "slice_frame_w_metadata",
                    "params:slice_events",
                    "params:eval_columns",
                ],
                outputs="slice_frame_filtered",
                name="filter_slices_of_slice_frame",
            ),
            Node(
                func=build_sampling_totals,
                inputs=["adoption_scenarios", "params:build_sampling_totals"],
                outputs="sampling_totals",
                name="build_sampling_totals",
            ),
            Node(
                func=start_dask_node,
                inputs="params:dask_eval",
                outputs=["dask_cluster_eval", "dask_client_eval"],
                name="start_dask_eval",
            ),
        ],
        tags="scenario_run",
    )

    report_profiles_scaled_pipe = Pipeline(
        [
            Node(
                func=build_eval_columns,
                inputs=["params:eval_columns", "params:group_columns"],
                outputs="eval_columns",
                name="build_eval_columns",
            ),
            Node(
                func=sample_vehicle_windows,
                inputs=[
                    "slices_filtered",
                    "slice_frame_filtered",
                    "sampling_totals",
                    "params:slice_events",
                    "params:sample_slices",
                    "eval_columns",
                    "dask_client_eval",
                ],
                outputs=["bootstrap_profiles", "report_by_region_energies"],
                name="sample_vehicle_windows",
            ),
            Node(
                func=summarize_vehicle_window_quantiles,
                inputs=[
                    "bootstrap_profiles",
                    "params:slice_events",
                    "params:summarize_slices",
                    "eval_columns",
                ],
                outputs="report_by_region_quantiles",
                name="summarize_vehicle_window_quantiles",
            ),
            Node(
                func=write_scenario_partition,
                inputs=["report_by_region_quantiles", "params:results_partition"],
                outputs="report_by_region_quantiles_partition",
                name="write_scenario_partition_hexes_quants",
            ),
            Node(
                func=write_scenario_partition,
                inputs=["report_by_region_energies", "params:results_partition"],
                outputs="report_by_region_energies_partition",
                name="write_scenario_partition_region_energies",
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
    }
    profile_group_fixed_inputs = {
        "slices_filtered",
        "slice_frame_filtered",
        "sampling_totals",
        "dask_client_eval",
    }

    report_profiles_pipes = [
        Pipeline(
            report_profiles_scaled_pipe,
            namespace="substation",
            parameters=profile_group_fixed_params,
            inputs=profile_group_fixed_inputs,
            tags="scenario_run",
        ),
        Pipeline(
            report_profiles_scaled_pipe,
            namespace="state_op_dist",
            parameters=profile_group_fixed_params,
            inputs=profile_group_fixed_inputs,
            tags="scenario_run",
        ),
    ]

    return (
        read_pipe
        + report_vehicles_pipe
        + report_profiles_scaled_prep_pipe
        + sum(report_profiles_pipes)
    )
