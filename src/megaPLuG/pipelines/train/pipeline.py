"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import eval_single_vars, get_posterior_predictive, train_pgm_full


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=train_pgm_full,
                inputs=[
                    "charging_sessions_simulated",
                    "params:pgm_struct",
                    "params:pgm_train",
                ],
                outputs="pgm_trained",
                name="train_pgm_full",
            ),
            node(
                func=get_posterior_predictive,
                inputs=["pgm_trained", "params:posterior_predictive"],
                outputs="posterior_predictive",
                name="get_posterior_predictive",
            ),
            node(
                func=eval_single_vars,
                inputs="pgm_trained",
                outputs="eval_plot_single_vars",
                name="eval_single_vars",
            ),
        ],
    )
    return pipe
