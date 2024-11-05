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
    add_region_geoms,
    assign_regions,
    assign_scale_up_factor,
    get_load_profiles,
    report_by_region_peaks,
    report_by_region_quantiles,
    summarize_vehicles,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
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
                func=assign_regions,
                inputs=["dwell_obj_eval", "hex_region_corresp"],
                outputs="dwell_obj_w_regions",
                name="assign_regions",
                tags="frame-charging_management",
            ),
            node(
                func=assign_scale_up_factor,
                inputs=[
                    "dwell_obj_w_regions",
                    "vehicles_labelled",
                    "params:assign_scale_up_factor",
                ],
                outputs="dwell_obj_w_scaling",
                name="assign_scale_up_factor",
                tags="frame-charging_management",
            ),
            node(
                func=get_load_profiles,
                inputs=["dwell_obj_w_scaling", "params:profiles_from_dwells"],
                outputs="profiles",
                name="get_load_profiles",
                tags="frame-charging_management",
            ),
            node(
                func=report_by_region_peaks,
                inputs=[
                    "profiles",
                    "hex_region_corresp",
                    "params:report_by_region_peaks",
                ],
                outputs="report_by_region_peaks",
                name="report_by_region_peaks",
                tags="frame-charging_management",
            ),
            node(
                func=report_by_region_quantiles,
                inputs=[
                    "profiles",
                    "hex_region_corresp",
                    "params:report_by_region_quantiles",
                ],
                outputs="report_by_region_quantiles",
                name="report_by_region_quantiles",
                tags="frame-charging_management",
            ),
            # From here down is saving out results
            node(
                func=write_scenario_partition,
                inputs=["vehicles_evaluated", "params:results_partition"],
                outputs="vehicles_evaluated_partition",
                name="write_scenario_partition_vehicles",
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
        tags="scenario_run",
    )

    geo_pipe = pipeline(
        [
            node(
                func=add_region_geoms,
                inputs=[
                    "report_by_region_peaks",
                    "hex_region_corresp",
                    "params:add_region_geoms",
                ],
                outputs="report_by_region_peaks_with_geoms",
                name="add_region_geoms",
            )
        ],
    )

    return pipe + geo_pipe
