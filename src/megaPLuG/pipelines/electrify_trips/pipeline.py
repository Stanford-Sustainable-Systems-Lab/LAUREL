"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set, save_dwell_set
from megaPLuG.scenarios.io import write_scenario_partition
from megaPLuG.utils.data import categorize_columns, filter_by_vals_in_cols
from megaPLuG.utils.params import set_entity_params

from .nodes import (
    apply_delays,
    calc_dwell_durations,
    calc_energy_use,
    filter_dwells,
    filter_vehicles,
    manage_charging,
    mark_critical_days,
    merge_dwellset_node,
    prepare_mode_loc_corresp,
    prepare_modes,
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
                func=categorize_columns,
                inputs="dwell_obj_filtered_vehs",
                outputs="dwell_obj_filtered_vehs_categorized",
                name="categorize_columns_electrify_trips",
            ),
            node(
                func=merge_dwellset_node,
                inputs=[
                    "dwell_obj_filtered_vehs_categorized",
                    "vehicles_with_params",
                    "params:merge_vehicle_params",
                ],
                outputs="dwell_obj_w_veh_params",
                name="merge_dwellset_node_veh_params",
            ),
            node(
                func=calc_dwell_durations,
                inputs=[
                    "dwell_obj_w_veh_params",
                    "params:calc_dwell_durations",
                ],
                outputs="dwell_obj_w_dur",
                name="calc_dwell_durations",
                tags="frame-spatiotemporal",
            ),
            node(
                func=prepare_modes,
                inputs="params:charging_modes",
                outputs="charging_modes",
                name="prepare_modes",
            ),
            node(
                func=prepare_mode_loc_corresp,
                inputs=["charging_modes", "params:mode_location_corresp"],
                outputs="mode_loc_corresp",
                name="prepare_mode_loc_corresp",
            ),
            node(
                func=merge_dwellset_node,
                inputs=[
                    "dwell_obj_w_dur",
                    "mode_loc_corresp",
                    "params:merge_avail_modes",
                ],
                outputs="dwell_obj_w_avail_modes",
                name="merge_dwellset_node_avail_modes",
            ),
            node(
                func=calc_energy_use,
                inputs=[
                    "dwell_obj_w_avail_modes",
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
                    "charging_modes",
                    "params:simulate_charging_choice",
                ],
                outputs="dwell_obj_w_charging",
                name="simulate_charging_choice",
                tags="frame-charging_choice",
            ),
            node(
                func=merge_dwellset_node,
                inputs=[
                    "dwell_obj_w_charging",
                    "charging_modes",
                    "params:merge_chosen_mode",
                ],
                outputs="dwell_obj_w_modes",
                name="merge_dwellset_node_chosen_mode",
            ),
            node(
                func=apply_delays,
                inputs=["dwell_obj_w_modes", "params:apply_delays"],
                outputs="dwell_obj_w_delays",
                name="apply_delays",
            ),
            node(
                func=manage_charging,
                inputs=[
                    "dwell_obj_w_delays",
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
