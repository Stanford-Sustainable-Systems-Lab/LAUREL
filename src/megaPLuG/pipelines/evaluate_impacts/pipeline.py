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

from .nodes import (
    assign_regions,
    assign_vehicle_metadata,
    get_load_profiles,
    report_by_region_peaks,
    report_by_region_quantiles,
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
        tags="report_vehicles",
    )

    report_profiles_pipe = pipeline(
        [
            node(
                func=assign_regions,
                inputs=["events_eval", "hex_region_corresp", "params:eval_columns"],
                outputs="events_w_regions",
                name="assign_regions_eval",
            ),
            node(
                func=assign_vehicle_metadata,
                inputs=[
                    "events_w_regions",
                    "vehicles_evaluated",
                    "params:eval_columns",
                ],
                outputs="events_w_metadata",
                name="assign_vehicle_metadata",
            ),
            node(
                func=get_load_profiles,
                inputs=[
                    "events_w_metadata",
                    "params:profiles_from_events",
                    "params:eval_columns",
                ],
                outputs="profiles",
                name="get_load_profiles",
            ),
            node(
                func=report_by_region_peaks,
                inputs=[
                    "profiles",
                    "params:report_by_region_peaks",
                    "params:eval_columns",
                ],
                outputs="report_by_region_peaks",
                name="report_by_region_peaks",
            ),
            node(
                func=report_by_region_quantiles,
                inputs=[
                    "profiles",
                    "params:report_by_region_quantiles",
                    "params:eval_columns",
                ],
                outputs="report_by_region_quantiles",
                name="report_by_region_quantiles",
            ),
            node(
                func=write_scenario_partition,
                inputs=["report_by_region_peaks", "params:results_partition"],
                outputs="report_by_region_peaks_partition",
                name="write_scenario_partition_hexes_peaks",
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
        "params:profiles_from_events",
        "params:results_partition",
        "params:report_by_region_peaks",
        "params:report_by_region_quantiles",
    }
    profile_group_fixed_inputs = {
        "events_eval",
        "hex_region_corresp",
        "vehicles_evaluated",
    }

    report_profiles_pipes = [
        pipeline(
            report_profiles_pipe,
            namespace="substation",
            parameters=profile_group_fixed_params,
            inputs=profile_group_fixed_inputs,
            tags="scenario_run",
        ),
        pipeline(
            report_profiles_pipe,
            namespace="land_op_segment",
            parameters=profile_group_fixed_params,
            inputs=profile_group_fixed_inputs,
            tags="scenario_run",
        ),
        pipeline(
            report_profiles_pipe,
            namespace="op_segment",
            parameters=profile_group_fixed_params,
            inputs=profile_group_fixed_inputs,
            tags="scenario_run",
        ),
    ]

    return read_pipe + report_vehicles_pipe + sum(report_profiles_pipes)
