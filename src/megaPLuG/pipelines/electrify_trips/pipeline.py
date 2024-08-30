"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set, save_dwell_set

from .nodes import (
    calc_energy_use,
    filter_vehicles,
    mark_critical_days,
    mark_locations,
    mark_vehicle_days,
    set_charging_availability,
    simulate_charging_choice,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=load_dwell_set,
                inputs=["dwells", "params:load_dwell_set"],
                outputs="dwell_obj",
                name="load_dwell_set",
            ),
            node(
                func=filter_vehicles,
                inputs=["dwell_obj", "vehicles_labelled"],
                outputs="dwell_obj_filtered",
                name="filter_vehicles",
            ),
            node(
                func=mark_locations,
                inputs=[
                    "dwell_obj_filtered",
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
                inputs=["dwell_obj_veh_days", "params:crit_days"],
                outputs="dwell_obj_crit_days",
                name="mark_critical_days",
            ),
            node(
                func=calc_energy_use,
                inputs=["dwell_obj_crit_days", "params:vehicles"],
                outputs="dwell_obj_w_energy",
                name="calc_energy_use",
            ),
            node(
                func=set_charging_availability,
                inputs=["dwell_obj_w_energy", "params:vehicles", "params:locations"],
                outputs="dwell_obj_w_avail",
                name="set_charging_availability",
            ),
            node(
                func=simulate_charging_choice,
                inputs=["dwell_obj_w_avail", "params:vehicles"],
                outputs="dwell_obj_w_charging",
                name="simulate_charging_choice",
            ),
            node(
                func=save_dwell_set,
                inputs="dwell_obj_w_charging",
                outputs="dwells_with_charging",
                name="save_dwell_set",
            ),
        ],
    )
    return pipe
