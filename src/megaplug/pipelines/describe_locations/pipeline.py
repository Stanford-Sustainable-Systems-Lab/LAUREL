"""
This is a boilerplate pipeline 'describe_locations'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Node, Pipeline

from megaplug.utils.data import (
    categorize_columns,
    filter_by_vals_in_cols,
    select_columns,
)
from megaplug.utils.distributed import load_in_memory_node
from megaplug.utils.time import get_timezones

from .nodes import (
    apply_groups,
    clip_to_extent,
    collapse_naics_classes,
    concat_columns,
    concat_extra_estabs,
    describe_substation_usage,
    fill_missingness,
    fill_out_substations,
    format_estabs,
    format_extra_estabs,
    format_highways,
    format_states,
    format_substation_boundaries_pg_and_e,
    format_substation_profiles,
    format_substations_contin,
    format_urban,
    get_osm_estabs_truck_stops,
    get_osm_estabs_warehouses,
    group_hexes,
    hexify_polygons,
    pivot_hex_estabs,
    pivot_hex_land_use,
    prepare_stop_locations_public,
    reassign_hqs,
)


def create_pipeline(**kwargs) -> Pipeline:
    ca_subs_pipe = Pipeline(
        [
            Node(
                func=format_substation_boundaries_pg_and_e,
                inputs=["substations_pg_and_e", "params:format_substations_pg_and_e"],
                outputs="substations_standard_pg_and_e",
                name="format_substation_geographies_ca",
            ),
            Node(
                func=format_substation_profiles,
                inputs=[
                    "substation_profiles_pg_and_e",
                    "params:format_substation_profiles",
                ],
                outputs="substation_profiles_formatted",
                name="format_substation_profiles",
            ),
            Node(
                func=describe_substation_usage,
                inputs=[
                    "substation_profiles_formatted",
                    "substations_standard_pg_and_e",
                    "params:describe_substation_usage",
                ],
                outputs="substation_usage",
                name="describe_substation_usage",
            ),
        ],
        tags="california_substations",
    )

    continental_subs_pipe = Pipeline(
        [
            Node(
                func=filter_by_vals_in_cols,
                inputs=["substations_continent", "params:filter_contin_subs"],
                outputs="substations_continent_select",
                name="filter_by_vals_in_cols_substations_contin",
            ),
            Node(
                func=format_substations_contin,
                inputs=[
                    "substations_continent_select",
                    "params:format_substations_contin",
                ],
                outputs="substations_continent_formatted",
                name="format_substation_geographies_contin",
            ),
            Node(
                func=fill_out_substations,
                inputs=[
                    "substations_standard_pg_and_e",
                    "substations_continent_formatted",
                    "params:fill_out_substations",
                ],
                outputs="substations_filled_out",
                name="fill_out_substations",
            ),
            Node(
                func=clip_to_extent,
                inputs=[
                    "substations_filled_out",
                    "states_formatted",
                    "params:clip_to_extent",
                ],
                outputs="substations_formatted",
                name="clip_to_extent_substations",
            ),
        ],
        tags="continental_substations",
    )

    extra_estabs_pipe = Pipeline(
        [
            Node(
                func=prepare_stop_locations_public,
                inputs=["parking_public", "params:prepare_stop_locations_public"],
                outputs="estabs_public_parking",
                name="prepare_stop_locations_public",
            ),
            Node(
                func=get_osm_estabs_truck_stops,
                inputs=["params:get_osm_estabs", "params:get_osm_estabs_truck_stops"],
                outputs="estabs_osm_truck_stops",
                name="get_osm_estabs_truck_stops",
            ),
            Node(
                func=get_osm_estabs_warehouses,
                inputs=["params:get_osm_estabs", "params:get_osm_estabs_warehouses"],
                outputs="estabs_osm_warehouses",
                name="get_osm_estabs_warehouses",
            ),
            Node(
                func=concat_extra_estabs,
                inputs=[
                    "estabs_public_parking",
                    "estabs_osm_truck_stops",
                    "estabs_osm_warehouses",
                ],
                outputs="establishments_extra",
                name="concat_extra_estabs",
            ),
            Node(
                func=format_extra_estabs,
                inputs=[
                    "establishments_extra",
                    "params:format_extra_estabs",
                ],
                # WARNING: Saving this out causes dtype issue when concatenating with the other establishments (uint64 vs. int64)
                outputs="establishments_extra_formatted",
                name="format_extra_estabs",
                tags=["fast_loc_grouping"],
            ),
        ],
        tags=["extra_estabs"],
    )

    estab_pipe = Pipeline(
        [
            Node(
                func=format_estabs,
                inputs=[
                    "establishments_name_naics_employees",
                    "establishments_location",
                    "establishments_parent_bus_status",
                    "params:format_estabs",
                ],
                outputs="establishments_formatted",
                name="format_estabs",
            ),
            Node(
                func=load_in_memory_node,
                inputs="establishments_formatted",
                outputs="estabs_raw",
                name="load_to_memory_estabs_raw",
                tags=["fast_loc_grouping"],
            ),
            Node(
                func=reassign_hqs,
                inputs=["estabs_raw", "params:reassign_hqs"],
                outputs="estabs_hqed",
                name="reassign_hqs",
                tags=["fast_loc_grouping"],
            ),
            Node(
                func=concat_extra_estabs,
                inputs=["estabs_hqed", "establishments_extra_formatted"],
                outputs="estabs_w_extras",
                name="concat_extra_estabs_to_main",
                tags=["fast_loc_grouping"],
            ),
            Node(
                func=collapse_naics_classes,
                inputs=[
                    "estabs_w_extras",
                    "naics_freight_intensive",
                    "params:collapse_naics_classes",
                ],
                outputs="estabs_leafed",
                name="collapse_naics_classes",
                tags=["fast_loc_grouping"],
            ),
        ],
        tags="establishments",
    )

    state_pipe = Pipeline(
        [
            Node(
                func=format_states,
                inputs=["state_boundaries", "params:format_states"],
                outputs="states_formatted_raw",
                name="format_states",
            ),
            Node(
                func=filter_by_vals_in_cols,
                inputs=["states_formatted_raw", "params:filter_state_codes"],
                outputs="states_formatted",
                name="filter_by_vals_in_cols_govt",
            ),
        ],
        tags="format_states",
    )

    highway_pipe = Pipeline(
        [
            Node(
                func=format_highways,
                inputs=["highways", "params:format_highways"],
                outputs="highways_formatted_unclipped",
                name="format_highways",
            ),
            Node(
                func=clip_to_extent,
                inputs=[
                    "highways_formatted_unclipped",
                    "states_formatted",
                    "params:clip_to_extent",
                ],
                outputs="highways_formatted",
                name="clip_to_extent_highways",
            ),
        ],
        tags="format_highways",
    )

    urban_areas_pipe = Pipeline(
        [
            Node(
                func=format_urban,
                inputs=["urban_areas", "params:format_urban"],
                outputs="urban_areas_formatted_unclipped",
                name="format_urban",
            ),
            Node(
                func=clip_to_extent,
                inputs=[
                    "urban_areas_formatted_unclipped",
                    "states_formatted",
                    "params:clip_to_extent",
                ],
                outputs="urban_areas_formatted",
                name="clip_to_extent_urban_areas",
            ),
        ],
        tags="format_urban_areas",
    )

    polys_to_hexes_pipe = Pipeline(
        [
            Node(
                func=select_columns,
                inputs=["polys_formatted", "params:select_columns"],
                outputs="polys_col_select",
                name="select_columns",
            ),
            Node(
                func=hexify_polygons,
                inputs=["polys_col_select", "params:hexify_polygons"],
                outputs="area_hexes_raw",
                name="hexify_polygons",
            ),
            Node(
                func=categorize_columns,
                inputs="area_hexes_raw",
                outputs="area_hexes",
                name="categorize_columns_hexes",
            ),
        ],
        tags="polys_to_hexes",
    )

    polys_to_hexes_pipe_fixed_params = {
        "params:hexify_polygons",
    }

    polys_to_hexes_pipes = [
        Pipeline(
            polys_to_hexes_pipe,
            namespace="states",
            parameters=polys_to_hexes_pipe_fixed_params,
            inputs={
                "polys_formatted": "states_formatted",
            },
            outputs={
                "area_hexes": "states_area_hexes",
            },
        ),
        Pipeline(
            polys_to_hexes_pipe,
            namespace="highways",
            parameters=polys_to_hexes_pipe_fixed_params,
            inputs={
                "polys_formatted": "highways_formatted",
            },
            outputs={
                "area_hexes": "highways_area_hexes",
            },
        ),
        Pipeline(
            polys_to_hexes_pipe,
            namespace="urban_areas",
            parameters=polys_to_hexes_pipe_fixed_params,
            inputs={
                "polys_formatted": "urban_areas_formatted",
            },
            outputs={
                "area_hexes": "urban_areas_area_hexes",
            },
        ),
        Pipeline(
            polys_to_hexes_pipe,
            namespace="substations",
            parameters=polys_to_hexes_pipe_fixed_params,
            inputs={
                "polys_formatted": "substations_formatted",
            },
            outputs={
                "area_hexes": "substations_area_hexes",
            },
        ),
    ]

    tz_pipe = Pipeline(
        [
            Node(
                func=get_timezones,
                inputs=["states_area_hexes", "params:get_timezones"],
                outputs="states_hexes_tz_raw",
                name="get_timezones",
            ),
            Node(
                func=categorize_columns,
                inputs="states_hexes_tz_raw",
                outputs="states_hexes_tz",
                name="categorize_columns_states_tz",
            ),
        ],
    )

    concat_pipe = Pipeline(
        [
            Node(
                func=concat_columns,
                inputs=[
                    "states_hexes_tz",
                    "highways_area_hexes",
                    "urban_areas_area_hexes",
                    "substations_area_hexes",
                ],
                outputs="hex_base_corresp_w_missing",
                name="concat_columns",
            ),
            Node(
                func=fill_missingness,
                inputs=["hex_base_corresp_w_missing", "params:fill_missingness"],
                outputs="hex_base_corresp",
                name="fill_missingness",
            ),
        ],
    )

    cluster_pipe = Pipeline(
        [
            Node(
                func=pivot_hex_estabs,
                inputs=["estabs_leafed", "params:pivot_hex_estabs"],
                outputs="estabs_pivoted",
                name="pivot_hex_estabs",
                tags=["fast_loc_grouping"],
            ),
            Node(
                func=pivot_hex_land_use,
                inputs=["hex_land_use", "params:pivot_hex_land_use"],
                outputs="land_use_pivoted",
                name="pivot_hex_land_use",
                tags=["fast_loc_grouping"],
            ),
            Node(
                func=group_hexes,
                inputs=["land_use_pivoted", "estabs_pivoted", "params:group_hexes"],
                outputs="hex_clusters",
                name="group_hexes",
                tags=["fast_loc_grouping"],
            ),
            Node(
                func=apply_groups,
                inputs=[
                    "hex_base_corresp",
                    "hex_clusters",
                    "params:apply_groups",
                ],
                outputs="hex_cluster_corresp",
                name="apply_groups",
            ),
        ],
        tags="establishments",
    )

    return (
        ca_subs_pipe
        + continental_subs_pipe
        + extra_estabs_pipe
        + estab_pipe
        + state_pipe
        + tz_pipe
        + urban_areas_pipe
        + highway_pipe
        + cluster_pipe
        + sum(polys_to_hexes_pipes)
        + concat_pipe
    )
