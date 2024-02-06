"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import simulate_electrified_trips


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=simulate_electrified_trips,
                inputs=["trips", "energy_consumption", "params:vehicles"],
                outputs="charging_sessions_simulated",
                name="simulate_electrified_trips",
            ),
        ],
    )
    return pipe
