"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    add_geometries,
    report_by_hex,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=report_by_hex,
                inputs=["events", "params:report_by_hex"],
                outputs="report_by_hex",
                name="report_by_hex",
            ),
            node(
                func=add_geometries,
                inputs=["report_by_hex", "params:add_geometries"],
                outputs="report_by_hex_with_geoms",
                name="add_geometries",
            ),
        ],
    )
    return pipe
