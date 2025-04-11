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
    build_substation_polygons,
    describe_substation_usage,
    format_govt_areas,
    format_highways,
    format_substation_boundaries_contin,
    format_substation_boundaries_pg_and_e,
    format_substation_profiles,
    format_urban,
    get_hexes_by_area,
)


def create_pipeline(**kwargs) -> Pipeline:
    ca_subs_pipe = pipeline(
        [
            node(
                func=format_substation_boundaries_pg_and_e,
                inputs=["substations_pg_and_e", "params:format_substations_pg_and_e"],
                outputs="pg_and_e.substations_standard",
                name="format_substation_geographies_ca",
            ),
            node(
                func=format_substation_profiles,
                inputs=[
                    "substation_profiles_pg_and_e",
                    "params:format_substation_profiles",
                ],
                outputs="substation_profiles_formatted",
                name="format_substation_profiles",
            ),
            node(
                func=describe_substation_usage,
                inputs=[
                    "substation_profiles_formatted",
                    "pg_and_e.substations_standard",
                    "params:describe_substation_usage",
                ],
                outputs="substation.usage",
                name="describe_substation_usage",
            ),
        ],
        tags="california_substations",
    )

    continental_subs_pipe = pipeline(
        [
            node(
                func=format_substation_boundaries_contin,
                inputs=["substations_continent", "params:format_substations_contin"],
                outputs="substations_continent_formatted",
                name="format_substation_geographies_contin",
            ),
            node(
                func=filter_by_vals_in_cols,
                inputs=[
                    "substations_continent_formatted",
                    "params:filter_contin_bounds",
                ],
                outputs="substations_continent_select",
                name="filter_by_vals_in_cols_substations_contin",
            ),
            node(
                func=filter_by_vals_in_cols,
                inputs=["states_formatted", "params:govt_areas_contin"],
                outputs="govt_areas_contin",
                name="filter_by_vals_in_cols_govt_contin",
            ),
            node(
                func=build_substation_polygons,
                inputs=[
                    "substations_continent_select",
                    "govt_areas_contin",
                    "params:build_substation_polygons",
                ],
                outputs="continental.substations_standard",
                name="build_substation_polygons",
            ),
        ],
        tags="continental_substations",
    )

    format_allied_pipe = pipeline(
        [
            node(
                func=format_govt_areas,
                inputs=["state_boundaries", "params:format_govt_areas"],
                outputs="states_formatted",
                name="format_govt_areas",
            ),
            node(
                func=format_urban,
                inputs=["urban_areas", "params:format_urban"],
                outputs="urban_areas_formatted",
                name="format_urban",
            ),
            node(
                func=format_highways,
                inputs=["highways", "states_formatted", "params:format_highways"],
                outputs="highways_formatted",
                name="format_highways",
            ),
        ],
        tags="format_allied_datasets",
    )

    geo_pipe = pipeline(
        [
            node(
                func=filter_by_vals_in_cols,
                inputs=["states_formatted", "params:filter_state_codes"],
                outputs="states_select",
                name="filter_by_vals_in_cols_govt",
            ),
            node(
                func=filter_by_vals_in_cols,
                inputs=["highways_formatted", "params:filter_state_codes"],
                outputs="highways_select",
                name="filter_by_vals_in_cols_highway",
            ),
            node(
                func=filter_by_vals_in_cols,
                inputs=["urban_areas_formatted", "params:filter_state_codes"],
                outputs="urban_areas_select",
                name="filter_urban_areas",
            ),
            node(
                func=filter_by_vals_in_cols,
                inputs=["substations_standard", "params:filter_state_codes"],
                outputs="substation_geographies_select",
                name="filter_by_vals_in_cols_substations",
            ),
            node(
                func=build_land_use_areas,
                inputs=[
                    "states_select",
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
                    "states_select",
                    "substation_geographies_select",
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
        ],
        tags="substation_geographies",
    )

    geo_pipe_fixed_params = {
        "params:land_use",
        "params:analysis_areas",
        "params:get_hexes_by_area",
        "params:get_timezones",
    }
    geo_pipe_fixed_inputs = {
        "states_formatted",
        "highways_formatted",
        "urban_areas_formatted",
    }

    geo_pipes = [
        pipeline(
            geo_pipe,
            namespace="pg_and_e",
            parameters=geo_pipe_fixed_params,
            inputs=geo_pipe_fixed_inputs,
            tags="geos_pg_and_e",
        ),
        pipeline(
            geo_pipe,
            namespace="continental",
            parameters=geo_pipe_fixed_params,
            inputs=geo_pipe_fixed_inputs,
            tags="geos_continental",
        ),
    ]

    return ca_subs_pipe + continental_subs_pipe + format_allied_pipe + sum(geo_pipes)
