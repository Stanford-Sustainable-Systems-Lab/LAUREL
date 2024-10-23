"""
This is a boilerplate pipeline 'describe_locations'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.utils.data import filter_by_vals_in_cols
from megaPLuG.utils.time import get_timezones

from .nodes import (
    build_analysis_areas,
    build_nearest_infra_corresp,
    build_utility_territory,
    get_hexes_by_area,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=build_utility_territory,
                inputs=["substations", "params:utility_territories"],
                outputs="territories",
                name="build_utility_territory",
            ),
            node(
                func=filter_by_vals_in_cols,
                inputs=["state_boundaries", "params:govt_areas"],
                outputs="govt_areas",
                name="filter_by_vals_in_cols",
            ),
            node(
                func=build_analysis_areas,
                inputs=["govt_areas", "territories"],
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
                func=build_nearest_infra_corresp,
                inputs=[
                    "hex_area_corresp",
                    "substations",
                    "params:nearest_infra_corresp",
                ],
                outputs="hex_infra_corresp",
                name="build_nearest_infra_corresp",
            ),
            node(
                func=get_timezones,
                inputs=["hex_infra_corresp", "params:get_timezones"],
                outputs="hex_region_corresp",
                name="get_timezones",
            ),
        ]
    )
    return pipe
