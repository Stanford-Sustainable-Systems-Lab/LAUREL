"""
This is a boilerplate pipeline 'describe_vehicles'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Node, Pipeline

from megaPLuG.models.dwell_sets import load_dwell_set, save_dwell_set
from megaPLuG.pipelines.electrify_trips.nodes import merge_dwellset_node

from .nodes import (
    calc_inter_visit_stats,
    calc_rolling_dwell_ratios,
    classify_vehicles,
    describe_veh_loc_pairs,
    filter_dwells_for_op_segment,
    filter_substantial_dwells,
    get_operating_segment,
    get_vehicle_observation_frames,
    group_veh_loc_pairs,
    label_veh_loc_pairs,
    mark_location_regions,
    mark_locations,
    mark_vehicle_centers,
    mark_weight_class_group,
    prepare_shared_locations,
)


def create_pipeline(**kwargs) -> Pipeline:
    veh_loc_pipe = Pipeline(
        [
            Node(
                func=load_dwell_set,
                inputs=["dwells", "params:load_dwell_set"],
                outputs="dwell_obj_desc_veh_locs",
                name="load_dwell_set_desc_veh_locs",
            ),
            Node(
                func=filter_substantial_dwells,
                inputs=["dwell_obj_desc_veh_locs", "params:substantial_dwells"],
                outputs="dwell_obj_filtered_desc_veh_locs",
                name="filter_substantial_dwells",
            ),
            Node(
                func=calc_inter_visit_stats,
                inputs="dwell_obj_filtered_desc_veh_locs",
                outputs="dwell_obj_inter_visit_desc_veh_locs",
                name="calc_inter_visit_stats",
            ),
            Node(
                func=calc_rolling_dwell_ratios,
                inputs=[
                    "dwell_obj_inter_visit_desc_veh_locs",
                    "params:rolling_dwell_ratios",
                ],
                outputs="dwell_obj_roll_desc_veh_locs",
                name="calc_rolling_dwell_ratios",
            ),
            Node(
                func=describe_veh_loc_pairs,
                inputs="dwell_obj_roll_desc_veh_locs",
                outputs="vehicle_location_pairs",
                name="describe_veh_loc_pairs",
            ),
            # Node(
            #     func=cluster_veh_loc_pairs,
            #     inputs=["vehicle_location_pairs", "params:cluster_veh_loc_pairs"],
            #     outputs="vehicle_location_pairs_clustered",
            #     name="cluster_veh_loc_pairs",
            # ),
            Node(
                func=group_veh_loc_pairs,
                inputs=["vehicle_location_pairs", "params:group_veh_loc_pairs"],
                outputs="vehicle_location_pairs_clustered",
                name="group_veh_loc_pairs",
            ),
            Node(
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

    veh_pipe = Pipeline(
        [
            Node(
                func=load_dwell_set,
                inputs=["dwells", "params:load_dwell_set"],
                outputs="dwell_obj_desc_vehs",
                name="load_dwell_set_desc_vehs",
            ),
            Node(
                func=mark_vehicle_centers,
                inputs=[
                    "vehicles_raw",
                    "vehicle_location_pairs_labelled",
                    "params:mark_vehicle_centers",
                    "params:load_dwell_set",
                ],
                outputs="vehicles_with_centers",
                name="mark_vehicle_centers",
            ),
            Node(
                func=filter_dwells_for_op_segment,
                inputs="dwell_obj_desc_vehs",
                outputs="dwell_obj_filtered_desc_vehs",
                name="filter_dwells_for_op_segment",
            ),
            Node(
                func=get_operating_segment,
                inputs=[
                    "vehicles_with_centers",
                    "dwell_obj_filtered_desc_vehs",
                    "params:operating_segment",
                ],
                outputs="vehicles_with_segment",
                name="get_operating_segment",
            ),
            Node(
                func=classify_vehicles,
                inputs=[
                    "vehicles_with_segment",
                    "vehicle_location_pairs_labelled",
                    "params:classify_vehicles",
                ],
                outputs="vehicles_with_class",
                name="classify_vehicles",
            ),
            Node(
                func=get_vehicle_observation_frames,
                inputs=[
                    "vehicles_with_class",
                    "dwell_obj_filtered_desc_vehs",
                    "params:observation_frames",
                ],
                outputs="vehicles_with_obs",
                name="get_vehicle_observation_frames",
            ),
            Node(
                func=mark_weight_class_group,
                inputs=[
                    "vehicles_with_obs",
                    "params:weight_class_group",
                ],
                outputs="vehs_with_weight_class_group",
                name="mark_weight_class_group",
            ),
            Node(
                func=mark_location_regions,
                inputs=[
                    "vehs_with_weight_class_group",
                    "state_boundaries",
                    "params:mark_location_regions",
                ],
                outputs="vehicles_labelled",
                name="mark_location_regions",
            ),
        ],
        tags="describe_vehs",
    )

    dwell_pipe = Pipeline(
        [
            Node(
                func=load_dwell_set,
                inputs=["dwells", "params:load_dwell_set"],
                outputs="dwell_obj_desc_dwells",
                name="load_dwell_set_desc_dwells",
            ),
            Node(
                func=prepare_shared_locations,
                inputs=["parking_formatted", "params:prepare_shared_locations"],
                outputs="shared_locations",
                name="prepare_shared_locations",
            ),
            Node(
                func=merge_dwellset_node,
                inputs=[
                    "dwell_obj_desc_dwells",
                    "shared_locations",
                    "params:merge_locations_shared",
                ],
                outputs="dwell_obj_desc_dwells_with_shared",
                name="merge_shared_locations",
            ),
            Node(
                func=merge_dwellset_node,
                inputs=[
                    "dwell_obj_desc_dwells_with_shared",
                    "vehicle_location_pairs_labelled",
                    "params:merge_locations_vehicle_specific",
                ],
                outputs="dwell_obj_desc_dwells_with_vehicle_specific",
                name="merge_vehicle_specific_locations",
            ),
            Node(
                func=mark_locations,
                inputs=[
                    "dwell_obj_desc_dwells_with_vehicle_specific",
                    "params:mark_locations",
                ],
                outputs="dwell_obj_with_locations",
                name="mark_locations",
            ),
            Node(
                func=save_dwell_set,
                inputs="dwell_obj_with_locations",
                outputs="dwells_with_locations",
                name="save_dwell_set_desc_vehs",
            ),
        ],
        tags="describe_dwells",
    )

    return veh_loc_pipe + veh_pipe + dwell_pipe
