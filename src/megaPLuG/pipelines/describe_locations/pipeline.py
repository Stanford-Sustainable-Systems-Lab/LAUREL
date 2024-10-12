"""
This is a boilerplate pipeline 'describe_locations'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set

from .nodes import build_substation_location_corresp


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=load_dwell_set,
                inputs=["dwells", "params:load_dwell_set"],
                outputs="dwell_obj_desc_locs",
                name="load_dwell_set_desc_locs",
            ),
            node(
                func=build_substation_location_corresp,
                inputs=[
                    "dwell_obj_desc_locs",
                    "substations",
                    "params:substation_location_corresp",
                ],
                outputs="hex_region_corresp",
                name="build_substation_location_corresp",
            ),
        ]
    )
    return pipe
