"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calc_dwell_hrs,
    calc_energy_use,
    set_charging_availability,
    simulate_charging_choice,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=calc_energy_use,
                inputs=["trips", "params:energy_use"],
                outputs="trips_with_energy",
                name="calc_energy_use",
            ),
            node(
                func=calc_dwell_hrs,
                inputs=["trips_with_energy", "params:dwell_times"],
                outputs="trips_with_dwells",
                name="calc_dwell_hrs",
            ),
            node(
                func=set_charging_availability,
                inputs="trips_with_dwells",
                outputs="trips_with_avail",
                name="set_charging_availability",
            ),
            node(
                func=simulate_charging_choice,
                inputs=["trips_with_avail", "params:charging_choice"],
                outputs="trips_with_charging",
                name="simulate_charging_choice",
            ),
        ],
    )
    return pipe
