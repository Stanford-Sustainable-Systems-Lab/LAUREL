"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set, save_dwell_set
from megaPLuG.scenarios.manage_scenarios import write_scenario_partition

from .nodes import (
    calc_energy_use,
    filter_dwells,
    filter_noncritical_dwells,
    filter_vehicles,
    get_hex_events_from_dwells,
    mark_critical_days,
    mark_locations,
    mark_vehicle_days,
    set_charging_availability,
    set_vehicle_params,
    simulate_charging_choice,
    summarize_vehicles,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=set_vehicle_params,
                inputs=[
                    "vehicles_labelled",
                    "params:vehicles",
                    "params:load_dwell_set",
                ],
                outputs="vehicles_with_params",
                name="set_vehicle_params",
            ),
            node(
                func=load_dwell_set,
                inputs=["dwells", "params:load_dwell_set"],
                outputs="dwell_obj",
                name="load_dwell_set",
            ),
            node(
                func=filter_vehicles,
                inputs=["dwell_obj", "vehicles_with_params"],
                outputs="dwell_obj_filtered_vehs",
                name="filter_vehicles",
            ),
            node(
                func=filter_dwells,
                inputs=[
                    "dwell_obj_filtered_vehs",
                    "params:locations",
                    "params:filter_dwells",
                ],
                outputs="dwell_obj_filtered_dwells",
                name="filter_dwells",
            ),
            node(
                func=mark_locations,
                inputs=[
                    "dwell_obj_filtered_dwells",
                    "vehicle_location_pairs_labelled",
                    "params:mark_locations",
                ],
                outputs="dwell_obj_marked_loc",
                name="mark_locations",
            ),
            node(
                func=mark_vehicle_days,
                inputs=["dwell_obj_marked_loc", "params:veh_days"],
                outputs="dwell_obj_veh_days",
                name="mark_vehicle_days",
            ),
            node(
                func=mark_critical_days,
                inputs=["dwell_obj_veh_days", "params:vehicles", "params:veh_days"],
                outputs="dwell_obj_crit_days",
                name="mark_critical_days",
            ),
            node(
                func=filter_noncritical_dwells,
                inputs=["dwell_obj_crit_days", "params:veh_days"],
                outputs="dwell_obj_crit_dwells",
                name="filter_noncritical_dwells",
            ),
            node(
                func=calc_energy_use,
                inputs=["dwell_obj_crit_dwells", "params:vehicles"],
                outputs="dwell_obj_w_energy",
                name="calc_energy_use",
            ),
            node(
                func=set_charging_availability,
                inputs=["dwell_obj_w_energy", "params:locations"],
                outputs="dwell_obj_w_avail",
                name="set_charging_availability",
            ),
            node(
                func=simulate_charging_choice,
                inputs=["dwell_obj_w_avail", "vehicles_with_params", "params:vehicles"],
                outputs="dwell_obj_w_charging",
                name="simulate_charging_choice",
            ),
            node(
                func=save_dwell_set,
                inputs="dwell_obj_w_charging",
                outputs="dwells_with_charging",
                name="save_dwell_set",
            ),
            node(
                func=write_scenario_partition,
                inputs=[
                    "dwells_with_charging",
                    "params:results_partition",
                ],
                outputs="dwells_with_charging_partition",
                name="write_scenario_partition_dwells",
            ),
            node(
                func=summarize_vehicles,
                inputs=[
                    "dwell_obj_w_charging",
                    "vehicles_with_params",
                    "params:summarize_vehicles",
                ],
                outputs="vehicles_evaluated",
                name="summarize_vehicles",
            ),
            node(
                func=get_hex_events_from_dwells,
                inputs=["dwell_obj_w_charging", "params:events_from_dwells"],
                outputs="events",
                name="get_hex_events_from_dwells",
            ),
            node(
                func=write_scenario_partition,
                inputs=[
                    "vehicles_evaluated",
                    "params:results_partition",
                ],
                outputs="vehicles_evaluated_partition",
                name="write_scenario_partition_vehicles",
            ),
        ],
    )
    return pipe
