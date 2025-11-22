"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Node, Pipeline

from megaplug.models.dwell_sets import load_dwell_set, save_dwell_set
from megaplug.scenarios.io import write_scenario_partition
from megaplug.utils.data import categorize_columns, filter_by_vals_in_cols
from megaplug.utils.distributed import start_dask_node
from megaplug.utils.params import set_entity_params

from .nodes import (
    assign_modes,
    calc_dwell_durations,
    calc_energy_use,
    calc_vehicle_ranges,
    filter_dwells,
    filter_dwells_post,
    filter_vehicles,
    mark_critical_days,
    mark_shift_powers,
    merge_dwellset_node,
    prepare_modes,
    simulate_charging_choice,
)


def create_pipeline(**kwargs) -> Pipeline:
    charge_pipe = Pipeline(
        [
            Node(
                func=start_dask_node,
                inputs=["params:dask_electrify_trips"],
                outputs=["dask_cluster_elect", "dask_client_elect"],
                name="start_dask_node_electrify_trips",
            ),
            Node(
                func=filter_by_vals_in_cols,
                inputs=["vehicles_labelled", "params:filter_vehicles"],
                outputs="vehicles_filtered",
                name="filter_by_vals_in_cols_vehs",
            ),
            Node(
                func=set_entity_params,
                inputs=["vehicles_filtered", "params:vehicles"],
                outputs="vehicles_with_params",
                name="set_entity_params_vehicles",
            ),
            Node(
                func=load_dwell_set,
                inputs=["dwells_with_locations_dask", "params:load_dwell_set"],
                outputs="dwell_obj",
                name="load_dwell_set_electrify_trips",
            ),
            Node(
                func=filter_vehicles,
                inputs=[
                    "dwell_obj",
                    "vehicles_with_params",
                    "params:dask_electrify_trips",
                ],
                outputs="dwell_obj_filtered_vehs",
                name="filter_vehicles",
            ),
            Node(
                func=categorize_columns,
                inputs="dwell_obj_filtered_vehs",
                outputs="dwell_obj_filtered_vehs_categorized",
                name="categorize_columns_electrify_trips",
            ),
            Node(
                func=calc_vehicle_ranges,
                inputs=[
                    "vehicles_with_params",
                    "dwell_obj",  # Using original dwell_obj to avoid double-computing
                    "params:calc_vehicle_ranges",
                ],
                outputs="vehicles_with_ranges",
                name="calc_vehicle_ranges",
            ),
            Node(
                func=merge_dwellset_node,
                inputs=[
                    "dwell_obj_filtered_vehs_categorized",
                    "vehicles_with_ranges",
                    "params:merge_vehicle_params",
                ],
                outputs="dwell_obj_w_veh_params",
                name="merge_dwellset_node_veh_params",
            ),
            Node(
                func=calc_dwell_durations,
                inputs=[
                    "dwell_obj_w_veh_params",
                    "params:calc_dwell_durations",
                ],
                outputs="dwell_obj_w_dur",
                name="calc_dwell_durations",
                tags="frame-spatiotemporal",
            ),
            Node(
                func=prepare_modes,
                inputs="params:charging_modes",
                outputs="charging_modes",
                name="prepare_modes",
            ),
            Node(
                func=assign_modes,
                inputs=["dwell_obj_w_dur", "charging_modes", "params:assign_modes"],
                outputs="dwell_obj_w_avail_modes",
                name="assign_modes",
            ),
            Node(
                func=calc_energy_use,
                inputs=[
                    "dwell_obj_w_avail_modes",
                    "params:calc_energy_use",
                ],
                outputs="dwell_obj_w_energy",
                name="calc_energy_use",
                tags="frame-energy",
            ),
            Node(
                func=mark_critical_days,
                inputs=[
                    "dwell_obj_w_energy",
                    "params:mark_critical_days",
                ],
                outputs="dwell_obj_crit_days",
                name="mark_critical_days",
                tags="frame-spatiotemporal",
            ),
            Node(
                func=filter_dwells,
                inputs=["dwell_obj_crit_days", "params:filter_dwells"],
                outputs="dwell_obj_crit_dwells",
                name="filter_dwells",
                tags="frame-spatiotemporal",
            ),
            Node(
                func=mark_shift_powers,
                inputs=["dwell_obj_crit_dwells", "params:mark_shift_powers"],
                outputs="dwell_obj_shift_powers",
                name="mark_shift_powers",
                tags="frame-charging_choice",
            ),
            Node(
                func=simulate_charging_choice,
                inputs=[
                    "dwell_obj_shift_powers",
                    "vehicles_with_params",
                    "charging_modes",
                    "params:simulate_charging_choice",
                ],
                outputs="dwell_obj_w_charging",
                name="simulate_charging_choice",
                tags="frame-charging_choice",
            ),
            Node(
                func=filter_dwells_post,
                inputs=["dwell_obj_w_charging", "params:filter_dwells_post"],
                outputs="dwell_obj_no_unused_optionals",
                name="filter_dwells_post",
                tags="frame-charging_choice",
            ),
            Node(
                func=merge_dwellset_node,
                inputs=[
                    "dwell_obj_no_unused_optionals",
                    "charging_modes",
                    "params:merge_chosen_mode",
                ],
                outputs="dwell_obj_w_modes",
                name="merge_dwellset_node_chosen_mode",
            ),
            Node(
                func=save_dwell_set,
                inputs="dwell_obj_w_modes",
                outputs="dwells_with_charging",
                name="save_dwell_set",
            ),
            Node(
                func=write_scenario_partition,
                inputs=["dwells_with_charging", "params:results_partition"],
                outputs="dwells_with_charging_partition_dask",
                name="write_scenario_partition_dwells",
            ),
            Node(
                func=write_scenario_partition,
                inputs=["vehicles_with_params", "params:results_partition"],
                outputs="vehicles_with_params_partition",
                name="write_scenario_partition_vehs",
            ),
        ],
        tags=["scenario_run", "choose_charging"],
    )

    return charge_pipe
