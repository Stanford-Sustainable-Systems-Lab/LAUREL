"""
This is a boilerplate pipeline 'build_scenario_runners'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .manage_cmd_calls import generate_bash_script
from .nodes import generate_scenario_configs


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=generate_scenario_configs,
                inputs="parameters",
                outputs="scenario_configs",
                name="generate_scenario_configs",
            ),
            node(
                func=generate_bash_script,
                inputs=[
                    "params:slurm_command",
                    "params:scenario_params",
                    "params:cmd_line_calls",
                    "scenario_configs",
                    "params:slurm_resources",
                    "params:slurm_reporting",
                ],
                outputs="scenario_slurm_script",
                name="generate_bash_script",
            ),
        ],
        tags="build_scenario_runners",
    )
    return pipe
