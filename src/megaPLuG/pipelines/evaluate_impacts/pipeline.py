"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Node, Pipeline

from megaPLuG.models.dwell_sets import load_dwell_set
from megaPLuG.pipelines.electrify_trips.nodes import (
    merge_dataframes_node,
    merge_dwellset_node,
)
from megaPLuG.scenarios.io import (
    read_scenario_partition,
    write_scenario_partition,
)
from megaPLuG.utils.data import (
    categorize_columns,
    get_merge_params,
)
from megaPLuG.utils.distributed import load_in_memory_node  # , start_dask_node

from .nodes import (
    add_dwell_id,
    apply_delays,
    build_class_frame,
    build_eval_columns,
    build_time_ordered_slice,
    compute_class_dwell_counts,
    compute_class_probs,
    compute_dwell_rate,
    compute_known_adoption_totals,
    filter_dwells_post_prob,
    filter_dwells_pre_prob,
    filter_locs_pre_prob,
    get_dwells_nonzero,
    localize_time_from_hexes,
    manage_charging,
    sample_profiles_node,
    slice_events,
    summarize_vehicle_window_quantiles,
    summarize_vehicles,
)


def create_pipeline(**kwargs) -> Pipeline:
    read_pipe = Pipeline(
        [
            Node(
                func=read_scenario_partition,
                inputs=[
                    "dwells_with_charging_partition_dask",
                    "params:results_partition",
                ],
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
                inputs=["hex_region_corresp_partition", "params:geo_partition_eval"],
                outputs="hex_region_corresp",
                name="collate_partitions_hexes",
            ),
            Node(
                func=filter_locs_pre_prob,
                inputs=["hex_region_corresp", "params:filter_locs_pre_prob"],
                outputs="hex_region_corresp_filtered",
                name="filter_locs_pre_prob",
            ),
            Node(
                func=categorize_columns,
                inputs="hex_region_corresp_filtered",
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
                    "dwell_obj_w_delays_mem",
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

    dwell_pipe_pre_prob = Pipeline(
        [
            Node(
                func=apply_delays,
                inputs=["dwell_obj_eval", "params:apply_delays"],
                outputs="dwell_obj_w_delays",
                name="apply_delays",
            ),
            Node(
                func=load_in_memory_node,
                inputs="dwell_obj_w_delays",
                outputs="dwell_obj_w_delays_mem",
                name="load_in_memory_eval",
            ),
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
                func=merge_dwellset_node,
                inputs=[
                    "dwell_obj_w_delays_mem",
                    "hex_region_corresp_categorized",
                    "merge_params_locations",
                ],
                outputs="dwell_obj_w_regions",
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
                func=merge_dwellset_node,
                inputs=[
                    "dwell_obj_w_regions",
                    "vehicles_evaluated",
                    "merge_params_vehicles",
                ],
                outputs="dwell_obj_w_metadata",
                name="assign_metadata_vehicles_to_events",
            ),
            Node(
                func=filter_dwells_pre_prob,
                inputs=[
                    "dwell_obj_w_metadata",
                    "params:filter_dwells_pre_prob_eval",
                    "params:eval_columns",
                ],
                outputs="dwell_obj_filtered",
                name="filter_dwells_pre_prob_eval",
            ),
        ],
        tags=["scenario_run"],
    )

    prob_pipe = Pipeline(
        [
            Node(
                func=build_class_frame,
                inputs=[
                    "hex_region_corresp_categorized",
                    "vehicles_evaluated",
                    "params:dwell_scaling",
                ],
                outputs="classes_frame",
                name="build_class_frame",
            ),
            Node(
                func=compute_class_dwell_counts,
                inputs=[
                    "classes_frame",
                    "dwell_obj_filtered",
                    "params:dwell_scaling",
                ],
                outputs="classes_counts",
                name="compute_class_dwell_counts",
            ),
            Node(
                func=compute_known_adoption_totals,
                inputs=[
                    "adoption_scenarios",
                    "params:compute_known_adoption_totals",
                    "params:dwell_scaling",
                ],
                outputs="veh_classes_adopt",
                name="compute_known_adoption_totals",
            ),
            Node(
                func=compute_dwell_rate,
                inputs=[
                    "veh_classes_adopt",
                    "dwell_obj_filtered",
                    "vehicles_evaluated",
                    "params:compute_dwell_rate",
                    "params:dwell_scaling",
                ],
                outputs="veh_classes_rate",
                name="compute_dwell_rate",
            ),
            Node(
                func=merge_dataframes_node,
                inputs=["classes_counts", "veh_classes_rate", "params:merge_classes"],
                outputs="classes_w_vehs",
                name="merge_classes",
            ),
            Node(
                func=compute_class_probs,
                inputs=[
                    "classes_w_vehs",
                    "params:compute_class_probs",
                ],
                outputs="classes_w_probs",
                name="compute_class_probs",
            ),
        ],
        tags=["scenario_run"],
    )

    dwell_pipe_post_prob = Pipeline(
        [
            Node(
                func=filter_dwells_post_prob,
                inputs=[
                    "dwell_obj_filtered",
                    "params:dwell_scaling",
                ],
                outputs="dwell_obj_electrified",
                name="filter_dwells_post_prob_eval",
            ),
            Node(
                func=merge_dwellset_node,
                inputs=[
                    "dwell_obj_electrified",
                    "classes_w_probs",
                    "params:merge_dwell_probs",
                ],
                outputs="dwell_obj_probabilities",
                name="merge_dwell_probs",
            ),
            Node(
                func=add_dwell_id,
                inputs=[
                    "dwell_obj_probabilities",
                    "params:eval_columns",
                ],
                outputs="dwell_obj_ided",
                name="add_dwell_id",
            ),
        ],
        tags=["scenario_run"],
    )

    event_pipe = Pipeline(
        [
            Node(
                func=get_dwells_nonzero,
                inputs=[
                    "dwell_obj_ided",
                    "params:eval_columns",
                ],
                outputs="dwell_obj_nonzero",
                name="get_dwells_nonzero",
            ),
            Node(
                func=manage_charging,
                inputs=[
                    "dwell_obj_nonzero",
                    "params:manage_charging",
                ],
                outputs="events",
                name="manage_charging",
                tags="frame-charging_management",
            ),
            Node(
                func=localize_time_from_hexes,
                inputs=[
                    "events",
                    "params:localize_time_from_hexes",
                    "params:eval_columns",
                ],
                outputs="events_w_local_time",
                name="localize_time_from_hexes",
            ),
            Node(
                func=slice_events,
                inputs=[
                    "events_w_local_time",
                    "params:slice_events",
                    "params:eval_columns",
                ],
                outputs="slices",
                name="slice_events",
            ),
            Node(
                func=build_time_ordered_slice,
                inputs=[
                    "slices",
                    "params:slice_events",
                    "params:eval_columns",
                ],
                outputs="slices_ordered",
                name="build_time_ordered_slice",
            ),
        ],
        tags=["scenario_run", "manage_charging"],
    )

    # report_profiles_scaled_prep_pipe = Pipeline(
    #     [
    #         Node(
    #             func=start_dask_node,
    #             inputs="params:dask_eval",
    #             outputs=["dask_cluster_eval", "dask_client_eval"],
    #             name="start_dask_eval",
    #         ),
    #     ],
    #     tags="scenario_run",
    # )

    report_profiles_scaled_pipe = Pipeline(
        [
            Node(
                func=build_eval_columns,
                inputs=["params:eval_columns", "params:group_columns"],
                outputs="eval_columns",
                name="build_eval_columns",
            ),
            Node(
                func=sample_profiles_node,
                inputs=[
                    "dwell_obj_ided",
                    "slices_ordered",
                    "hex_region_corresp_categorized",
                    "classes_w_probs",
                    "params:sample_profiles",
                    "eval_columns",
                    # "dask_client_eval",
                ],
                outputs="bootstrap_profiles",  # ["bootstrap_profiles", "report_by_region_energies"],
                name="sample_profiles_node",
            ),
            Node(
                func=summarize_vehicle_window_quantiles,
                inputs=[
                    "bootstrap_profiles",
                    "params:sample_profiles",
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
            # Node(
            #     func=write_scenario_partition,
            #     inputs=["report_by_region_energies", "params:results_partition"],
            #     outputs="report_by_region_energies_partition",
            #     name="write_scenario_partition_region_energies",
            # ),
        ],
        tags="report_profiles",
    )

    profile_group_fixed_params = {
        "params:results_partition",
        "params:eval_columns",
        "params:sample_profiles",
    }
    profile_group_fixed_inputs = {
        "dwell_obj_ided",
        "slices_ordered",
        "hex_region_corresp_categorized",
        "classes_w_probs",
        # "dask_client_eval",
    }

    report_profiles_pipes = [
        Pipeline(
            report_profiles_scaled_pipe,
            namespace="substation",
            parameters=profile_group_fixed_params,
            inputs=profile_group_fixed_inputs,
            tags="scenario_run",
        ),
        # Pipeline(
        #     report_profiles_scaled_pipe,
        #     namespace="state_op_dist",
        #     parameters=profile_group_fixed_params,
        #     inputs=profile_group_fixed_inputs,
        #     tags="scenario_run",
        # ),
    ]

    return (
        read_pipe
        + report_vehicles_pipe
        + dwell_pipe_pre_prob
        + prob_pipe
        + dwell_pipe_post_prob
        + event_pipe
        # + report_profiles_scaled_prep_pipe
        + sum(report_profiles_pipes)
    )
