"""
This is a boilerplate pipeline 'describe_vehicles'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Node, Pipeline

from megaplug.models.dwell_sets import load_dwell_set

from .nodes import (
    classify_vehicles,
    filter_dwells_for_op_segment,
    get_operating_segment,
    get_vehicle_observation_frames,
    mark_location_regions,
    mark_vehicle_centers,
    mark_weight_class_group,
    strip_vehicle_attrs,
)


def create_pipeline(**kwargs) -> Pipeline:
    veh_pipe = Pipeline(
        [
            Node(
                func=load_dwell_set,
                inputs=["dwells", "params:load_dwell_set"],
                outputs="dwell_obj_desc_vehs",
                name="load_dwell_set_desc_vehs",
            ),
            Node(
                func=strip_vehicle_attrs,
                inputs=["trips_formatted", "params:strip_vehicle_attrs"],
                outputs="vehicles_raw",
                name="strip_vehicle_attrs",
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

    return veh_pipe
