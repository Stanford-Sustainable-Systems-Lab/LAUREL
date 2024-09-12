"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set
from megaPLuG.scenarios.manage_scenarios import write_scenario_partition

from .nodes import (
    add_geometries,
    get_hex_events_from_dwells,
    report_by_hex,
    summarize_vehicles,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=load_dwell_set,
                inputs=["dwells_with_charging", "params:load_dwell_set"],
                outputs="dwell_obj_eval",
                name="load_dwell_set_eval_impacts",
            ),
            node(
                func=summarize_vehicles,
                inputs=[
                    "dwell_obj_eval",
                    "vehicles_with_params",
                    "params:summarize_vehicles",
                ],
                outputs="vehicles_evaluated",
                name="summarize_vehicles",
            ),
            node(
                func=get_hex_events_from_dwells,
                inputs=["dwell_obj_eval", "params:events_from_dwells"],
                outputs="events",
                name="get_hex_events_from_dwells",
            ),
            node(
                func=report_by_hex,
                inputs=["events", "params:report_by_hex"],
                outputs="report_by_hex",
                name="report_by_hex",
            ),
            node(
                func=add_geometries,
                inputs=["report_by_hex", "params:add_geometries"],
                outputs="report_by_hex_with_geoms",
                name="add_geometries",
            ),
            # From here down is saving out results
            node(
                func=write_scenario_partition,
                inputs=[
                    "vehicles_evaluated",
                    "params:results_partition",
                ],
                outputs="vehicles_evaluated_partition",
                name="write_scenario_partition_vehicles",
            ),
        ],
    )
    return pipe
