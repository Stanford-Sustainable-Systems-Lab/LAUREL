"""
This is a boilerplate pipeline 'sample_charging'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import calc_model_conditioning, sample_charging_sessions


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=calc_model_conditioning,
                inputs=["adoption_scenario", "vius_clean", "params:condition"],
                outputs="pgm_conditioning",
                name="calc_model_conditioning",
            ),
            node(
                func=sample_charging_sessions,
                inputs=["pgm_trained", "pgm_conditioning", "params:sample"],
                outputs="charging_sessions_sampled",
                name="sample_charging_sessions",
            ),
        ],
    )
    return pipe
