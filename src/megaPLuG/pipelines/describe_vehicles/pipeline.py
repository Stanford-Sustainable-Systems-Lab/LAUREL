"""
This is a boilerplate pipeline 'describe_vehicles'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set, save_dwell_set

from .nodes import (
    calc_inter_visit_stats,
    calc_rolling_dwell_ratios,
    classify_vehicles,
    describe_veh_loc_pairs,
    filter_substantial_dwells,
    get_operating_segment,
    get_vehicle_observation_frames,
    group_veh_loc_pairs,
    label_veh_loc_pairs,
    mark_location_regions,
    mark_locations,
    mark_weight_class_group,
)


def create_pipeline(**kwargs) -> Pipeline:
    veh_loc_pipe = pipeline(
        [
            node(
                func=load_dwell_set,
                inputs=["dwells", "params:load_dwell_set"],
                outputs="dwell_obj_desc_veh_locs",
                name="load_dwell_set_desc_veh_locs",
            ),
            node(
                func=filter_substantial_dwells,
                inputs=["dwell_obj_desc_veh_locs", "params:substantial_dwells"],
                outputs="dwell_obj_filtered_desc_veh_locs",
                name="filter_substantial_dwells",
            ),
            node(
                func=calc_inter_visit_stats,
                inputs="dwell_obj_filtered_desc_veh_locs",
                outputs="dwell_obj_inter_visit_desc_veh_locs",
                name="calc_inter_visit_stats",
            ),
            node(
                func=calc_rolling_dwell_ratios,
                inputs=[
                    "dwell_obj_inter_visit_desc_veh_locs",
                    "params:rolling_dwell_ratios",
                ],
                outputs="dwell_obj_roll_desc_veh_locs",
                name="calc_rolling_dwell_ratios",
            ),
            node(
                func=describe_veh_loc_pairs,
                inputs="dwell_obj_roll_desc_veh_locs",
                outputs="vehicle_location_pairs",
                name="describe_veh_loc_pairs",
            ),
            # node(
            #     func=cluster_veh_loc_pairs,
            #     inputs=["vehicle_location_pairs", "params:cluster_veh_loc_pairs"],
            #     outputs="vehicle_location_pairs_clustered",
            #     name="cluster_veh_loc_pairs",
            # ),
            node(
                func=group_veh_loc_pairs,
                inputs=["vehicle_location_pairs", "params:group_veh_loc_pairs"],
                outputs="vehicle_location_pairs_clustered",
                name="group_veh_loc_pairs",
            ),
            node(
                func=label_veh_loc_pairs,
                inputs=[
                    "vehicle_location_pairs_clustered",
                    "params:label_veh_loc_pairs",
                ],
                outputs="vehicle_location_pairs_labelled",
                name="label_veh_loc_pairs",
            ),
        ],
        tags="describe_veh_loc",
    )

    veh_pipe = pipeline(
        [
            node(
                func=load_dwell_set,
                inputs=["dwells", "params:load_dwell_set"],
                outputs="dwell_obj_desc_vehs",
                name="load_dwell_set_desc_vehs",
            ),
            node(
                func=classify_vehicles,
                inputs=[
                    "vehicles_raw",
                    "vehicle_location_pairs_labelled",
                    "params:classify_vehicles",
                ],
                outputs="vehicles_with_class",
                name="classify_vehicles",
            ),
            node(
                func=get_operating_segment,
                inputs=[
                    "vehicles_with_class",
                    "dwell_obj_desc_vehs",
                    "params:operating_segment",
                ],
                outputs="vehicles_with_segment",
                name="get_operating_segment",
            ),
            node(
                func=get_vehicle_observation_frames,
                inputs=[
                    "vehicles_with_segment",
                    "dwell_obj_desc_vehs",
                    "params:observation_frames",
                ],
                outputs="vehicles_with_obs",
                name="get_vehicle_observation_frames",
            ),
            node(
                func=mark_weight_class_group,
                inputs=[
                    "vehicles_with_obs",
                    "params:weight_class_group",
                ],
                outputs="vehs_with_weight_class_group",
                name="mark_weight_class_group",
            ),
            node(
                func=mark_location_regions,
                inputs=[
                    "vehs_with_weight_class_group",
                    "vehicle_location_pairs_labelled",
                    "state_boundaries",
                    "params:mark_location_regions",
                    "params:load_dwell_set",
                ],
                outputs="vehicles_labelled",
                name="mark_location_regions",
            ),
        ],
        tags="describe_vehs",
    )

    dwell_pipe = pipeline(
        [
            node(
                func=load_dwell_set,
                inputs=["dwells", "params:load_dwell_set"],
                outputs="dwell_obj_desc_dwells",
                name="load_dwell_set_desc_dwells",
            ),
            node(
                func=mark_locations,
                inputs=[
                    "dwell_obj_desc_dwells",
                    "vehicle_location_pairs_labelled",
                    "params:mark_locations",
                ],
                outputs="dwell_obj_with_locations",
                name="mark_locations",
            ),
            node(
                func=save_dwell_set,
                inputs="dwell_obj_with_locations",
                outputs="dwells_with_locations",
                name="save_dwell_set_desc_vehs",
            ),
        ],
        tags="describe_dwells",
    )

    return veh_loc_pipe + veh_pipe + dwell_pipe
