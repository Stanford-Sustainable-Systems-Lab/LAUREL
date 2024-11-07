"""
This is a boilerplate pipeline 'describe_vehicles'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set, save_dwell_set

from .nodes import (
    calc_inter_visit_stats,
    calc_vehicle_scaling_weights,
    classify_vehicles,
    describe_veh_loc_pairs,
    filter_substantial_dwells,
    get_operating_segment,
    label_veh_loc_pairs,
    mark_location_regions,
    mark_locations,
    mark_weight_class_group,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=load_dwell_set,
                inputs=["dwells", "params:load_dwell_set"],
                outputs="dwell_obj_desc_vehs",
                name="load_dwell_set_desc_vehs",
            ),
            node(
                func=filter_substantial_dwells,
                inputs=["dwell_obj_desc_vehs", "params:substantial_dwells"],
                outputs="dwell_obj_filtered_desc_vehs",
                name="filter_substantial_dwells",
            ),
            node(
                func=calc_inter_visit_stats,
                inputs="dwell_obj_filtered_desc_vehs",
                outputs="dwell_obj_inter_visit_desc_vehs",
                name="calc_inter_visit_stats",
            ),
            node(
                func=describe_veh_loc_pairs,
                inputs="dwell_obj_inter_visit_desc_vehs",
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
                func=label_veh_loc_pairs,
                inputs=[
                    "vehicle_location_pairs_clustered",
                    "params:label_veh_loc_pairs",
                ],
                outputs="vehicle_location_pairs_labelled",
                name="label_veh_loc_pairs",
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
                func=mark_locations,
                inputs=[
                    "dwell_obj_desc_vehs",
                    "vehicle_location_pairs_labelled",
                    "params:mark_locations",
                ],
                outputs="dwell_obj_with_locations_desc_vehs",
                name="mark_locations",
            ),
            node(
                func=mark_weight_class_group,
                inputs=[
                    "vehicles_with_segment",
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
                outputs="vehs_with_regions",
                name="mark_location_regions",
            ),
            node(
                func=calc_vehicle_scaling_weights,
                inputs=[
                    "vehs_with_regions",
                    "vius_scaling",
                    "params:vehicle_scaling_weights",
                ],
                outputs="vehicles_labelled",
                name="calc_vehicle_scaling_weights",
            ),
            node(
                func=save_dwell_set,
                inputs="dwell_obj_with_locations_desc_vehs",
                outputs="dwells_with_locations",
                name="save_dwell_set_desc_vehs",
            ),
        ]
    )
    return pipe
