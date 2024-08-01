"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set, save_dwell_set

from .nodes import (
    calc_energy_use,
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
                func=calc_energy_use,
                inputs=["dwell_obj", "params:vehicles"],
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
