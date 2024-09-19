"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set
from megaPLuG.scenarios.io import (
    read_scenario_partition,
    write_scenario_partition,
)

from .nodes import (
    get_load_profiles,
    report_by_hex,
    summarize_vehicles,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=read_scenario_partition,
                inputs=["dwells_with_charging_partition", "params:results_partition"],
                outputs="dwells_with_charging_eval",
                name="collate_partitions_dwells_with_charging",
            ),
            node(
                func=load_dwell_set,
                inputs=["dwells_with_charging_eval", "params:load_dwell_set"],
                outputs="dwell_obj_eval",
                name="load_dwell_set_eval_impacts",
            ),
            node(
                func=read_scenario_partition,
                inputs=["vehicles_with_params_partition", "params:results_partition"],
                outputs="vehicles_with_params_eval",
                name="collate_partitions_vehicles_with_params",
            ),
            node(
                func=summarize_vehicles,
                inputs=[
                    "dwell_obj_eval",
                    "vehicles_with_params_eval",
                    "params:summarize_vehicles",
                ],
                outputs="vehicles_evaluated",
                name="summarize_vehicles",
            ),
            node(
                func=get_load_profiles,
                inputs=["dwell_obj_eval", "params:profiles_from_dwells"],
                outputs="profiles",
                name="get_load_profiles",
            ),
            node(
                func=report_by_hex,
                inputs=["profiles", "params:report_by_hex"],
                outputs="report_by_hex",
                name="report_by_hex",
            ),
            # From here down is saving out results
            node(
                func=write_scenario_partition,
                inputs=["vehicles_evaluated", "params:results_partition"],
                outputs="vehicles_evaluated_partition",
                name="write_scenario_partition_vehicles",
            ),
            node(
                func=write_scenario_partition,
                inputs=["report_by_hex", "params:results_partition"],
                outputs="report_by_hex_partition",
                name="write_scenario_partition_hexes",
            ),
        ],
        tags="scenario_run",
    )
    return pipe
