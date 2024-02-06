"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_navistar


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=preprocess_navistar,
                inputs=["navistar", "params:navistar"],
                outputs="trips",
                name="preprocess_navistar",
            )
        ],
    )
    return pipe
