"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import read_navistar


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=read_navistar,
                inputs="navistar",
                outputs="test",
                name="read_navistar",
            )
        ],
    )
    return pipe
