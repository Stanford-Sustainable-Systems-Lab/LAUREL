"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set, save_dwell_set
from megaPLuG.scenarios.io import write_scenario_partition

from .nodes import (
    calc_energy_use,
    filter_dwells,
    filter_vehicles,
    mark_critical_days,
    mark_substantial_dwells,
    set_charging_availability,
    set_vehicle_params,
    simulate_charging_choice,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=set_vehicle_params,
                inputs=["vehicles_labelled", "params:vehicles"],
                outputs="vehicles_with_params",
                name="set_vehicle_params",
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
                func=set_charging_availability,
                inputs=["dwell_obj_filtered_vehs", "params:locations"],
                outputs="dwell_obj_w_avail",
                name="set_charging_availability",
            ),
            node(
                func=mark_substantial_dwells,
                inputs=[
                    "dwell_obj_w_avail",
                    "vehicles_with_params",
                    "params:mark_substantial_dwells",
                ],
                outputs="dwell_obj_filtered_dwells",
                name="mark_substantial_dwells",
            ),
            node(
                func=mark_critical_days,
                inputs=[
                    "dwell_obj_filtered_dwells",
                    "vehicles_with_params",
                    "params:mark_critical_days",
                ],
                outputs="dwell_obj_crit_days",
                name="mark_critical_days",
            ),
            node(
                func=filter_dwells,
                inputs=["dwell_obj_crit_days", "params:filter_dwells"],
                outputs="dwell_obj_crit_dwells",
                name="filter_dwells",
            ),
            node(
                func=calc_energy_use,
                inputs=[
                    "dwell_obj_crit_dwells",
                    "vehicles_with_params",
                    "params:calc_energy_use",
                ],
                outputs="dwell_obj_w_energy",
                name="calc_energy_use",
            ),
            node(
                func=simulate_charging_choice,
                inputs=[
                    "dwell_obj_w_energy",
                    "vehicles_with_params",
                    "params:simulate_charging_choice",
                ],
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
                inputs=["dwells_with_charging", "params:results_partition"],
                outputs="dwells_with_charging_partition",
                name="write_scenario_partition_dwells",
            ),
            node(
                func=write_scenario_partition,
                inputs=["vehicles_with_params", "params:results_partition"],
                outputs="vehicles_with_params_partition",
                name="write_scenario_partition_vehs",
            ),
        ],
    )
    return pipe
