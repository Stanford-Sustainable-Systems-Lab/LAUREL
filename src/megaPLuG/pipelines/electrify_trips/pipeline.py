"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calc_dwell_hrs,
    calc_energy_use,
    convert_to_pandas,
    simulate_charging_choice,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=convert_to_pandas,
                inputs=["trips"],
                outputs="trips_feather",
                name="convert_to_pandas",
            ),
            node(
                func=calc_energy_use,
                inputs=["trips_feather", "params:energy_use"],
                outputs="trips_with_energy",
                name="calc_energy_use",
            ),
            node(
                func=calc_dwell_hrs,
                inputs="trips_with_energy",
                outputs="trips_with_dwells",
                name="calc_dwell_hrs",
            ),
            node(
                func=simulate_charging_choice,
                inputs=["trips_with_dwells", "params:charging_choice"],
                outputs="trips_with_charging",
                name="simulate_charging_choice",
            ),
        ],
    )
    return pipe
