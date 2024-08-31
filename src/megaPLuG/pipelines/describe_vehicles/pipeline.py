"""
This is a boilerplate pipeline 'describe_vehicles'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set

from .nodes import (
    calc_inter_visit_stats,
    classify_vehicles,
    cluster_veh_loc_pairs,
    describe_veh_loc_pairs,
    filter_substantial_dwells,
    label_veh_loc_pairs,
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
            node(
                func=cluster_veh_loc_pairs,
                inputs=["vehicle_location_pairs", "params:cluster_veh_loc_pairs"],
                outputs="vehicle_location_pairs_clustered",
                name="cluster_veh_loc_pairs",
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
            node(
                func=classify_vehicles,
                inputs=[
                    "vehicles_raw",
                    "vehicle_location_pairs_labelled",
                    "params:classify_vehicles",
                ],
                outputs="vehicles_labelled",
                name="classify_vehicles",
            ),
        ]
    )
    return pipe
