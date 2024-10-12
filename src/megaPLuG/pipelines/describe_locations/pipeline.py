"""
This is a boilerplate pipeline 'describe_locations'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set
from megaPLuG.utils.time import get_timezones

from .nodes import build_substation_location_corresp, get_hex_geoms


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
                func=get_hex_geoms,
                inputs=["dwell_obj_desc_locs", "params:get_hex_geoms"],
                outputs="hex_geoms",
                name="get_hex_geoms",
            ),
            node(
                func=build_substation_location_corresp,
                inputs=[
                    "hex_geoms",
                    "substations",
                    "params:substation_location_corresp",
                ],
                outputs="hex_region_corresp_raw",
                name="build_substation_location_corresp",
            ),
            node(
                func=get_timezones,
                inputs=["hex_region_corresp_raw", "params:get_timezones"],
                outputs="hex_region_corresp",
                name="get_timezones",
            ),
        ]
    )
    return pipe
