"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set, save_dwell_set
from megaPLuG.scenarios.io import write_scenario_partition
from megaPLuG.utils.data import filter_by_vals_in_cols
from megaPLuG.utils.params import set_entity_params

from .nodes import (
    assign_regions,
    assign_scale_up_factor,
    calc_energy_use,
    filter_dwells,
    filter_vehicles,
    manage_charging,
    mark_critical_days,
    simulate_charging_choice,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=filter_by_vals_in_cols,
                inputs=["vehicles_labelled", "params:filter_vehicles"],
                outputs="vehicles_filtered",
                name="filter_by_vals_in_cols_vehs",
            ),
            node(
                func=set_entity_params,
                inputs=["vehicles_filtered", "params:vehicles"],
                outputs="vehicles_with_params",
                name="set_entity_params_vehicles",
            ),
            node(
                func=load_dwell_set,
                inputs=["dwells_with_locations", "params:load_dwell_set"],
                outputs="dwell_obj",
                name="load_dwell_set_electrify_trips",
            ),
            node(
                func=filter_vehicles,
                inputs=["dwell_obj", "vehicles_with_params"],
                outputs="dwell_obj_filtered_vehs",
                name="filter_vehicles",
            ),
            node(
                func=set_entity_params,
                inputs=["dwell_obj_filtered_vehs", "params:locations"],
                outputs="dwell_obj_w_avail",
                name="set_entity_params_locations",
                tags="frame-spatiotemporal",
            ),
            node(
                func=calc_energy_use,
                inputs=[
                    "dwell_obj_w_avail",
                    "vehicles_with_params",
                    "params:calc_energy_use",
                ],
                outputs="dwell_obj_w_energy",
                name="calc_energy_use",
                tags="frame-energy",
            ),
            node(
                func=mark_critical_days,
                inputs=[
                    "dwell_obj_w_energy",
                    "vehicles_with_params",
                    "params:mark_critical_days",
                ],
                outputs="dwell_obj_crit_days",
                name="mark_critical_days",
                tags="frame-spatiotemporal",
            ),
            node(
                func=filter_dwells,
                inputs=["dwell_obj_crit_days", "params:filter_dwells"],
                outputs="dwell_obj_crit_dwells",
                name="filter_dwells",
                tags="frame-spatiotemporal",
            ),
            node(
                func=simulate_charging_choice,
                inputs=[
                    "dwell_obj_crit_dwells",
                    "vehicles_with_params",
                    "params:charging_modes",
                    "params:simulate_charging_choice",
                ],
                outputs="dwell_obj_w_charging",
                name="simulate_charging_choice",
                tags="frame-charging_choice",
            ),
            node(
                func=assign_regions,
                inputs=["dwell_obj_w_charging", "hex_region_corresp"],
                outputs="dwell_obj_w_regions",
                name="assign_regions_electrify",
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
                func=manage_charging,
                inputs=[
                    "dwell_obj_w_scaling",
                    "params:manage_charging",
                ],
                outputs="events",
                name="manage_charging",
                tags="frame-charging_management",
            ),
            node(
                func=save_dwell_set,
                inputs="dwell_obj_w_charging",
                outputs="dwells_with_charging",
                name="save_dwell_set",
            ),
            node(
                func=write_scenario_partition,
                inputs=["dwells_with_charging", "params:results_partition"],
                outputs="dwells_with_charging_partition",
                name="write_scenario_partition_dwells",
            ),
            node(
                func=write_scenario_partition,
                inputs=["events", "params:results_partition"],
                outputs="events_partition",
                name="write_scenario_partition_events",
            ),
            node(
                func=write_scenario_partition,
                inputs=["vehicles_with_params", "params:results_partition"],
                outputs="vehicles_with_params_partition",
                name="write_scenario_partition_vehs",
            ),
        ],
        tags="scenario_run",
    )
    return pipe
