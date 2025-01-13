"""
This is a boilerplate pipeline 'describe_locations'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.utils.data import filter_by_vals_in_cols
from megaPLuG.utils.time import get_timezones

from .nodes import (
    build_analysis_areas_node,
    build_land_use_areas,
    filter_urban_areas,
    get_hexes_by_area,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=filter_by_vals_in_cols,
                inputs=["state_boundaries", "params:govt_areas"],
                outputs="govt_areas",
                name="filter_by_vals_in_cols_govt",
            ),
            node(
                func=filter_by_vals_in_cols,
                inputs=["highways", "params:highways"],
                outputs="highways_select",
                name="filter_by_vals_in_cols_highway",
            ),
            node(
                func=filter_urban_areas,
                inputs=["urban_areas", "params:urban_areas"],
                outputs="urban_areas_select",
                name="filter_urban_areas",
            ),
            node(
                func=build_land_use_areas,
                inputs=[
                    "govt_areas",
                    "highways_select",
                    "urban_areas_select",
                    "params:land_use",
                ],
                outputs="land_use",
                name="build_land_use_areas",
            ),
            node(
                func=build_analysis_areas_node,
                inputs=[
                    "govt_areas",
                    "substation_boundaries_formatted",
                    "land_use",
                    "params:analysis_areas",
                ],
                outputs="analysis_areas",
                name="build_analysis_areas",
            ),
            node(
                func=get_hexes_by_area,
                inputs=["analysis_areas", "params:get_hexes_by_area"],
                outputs="hex_area_corresp",
                name="get_hexes_by_area",
            ),
            node(
                func=get_timezones,
                inputs=["hex_area_corresp", "params:get_timezones"],
                outputs="hex_region_corresp",
                name="get_timezones",
            ),
        ]
    )
    return pipe
