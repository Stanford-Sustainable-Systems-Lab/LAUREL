"""
This is a boilerplate pipeline 'build_runners'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.scenarios.build import generate_scenario_configs
from megaPLuG.scenarios.cmd import generate_bash_script


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=generate_scenario_configs,
                inputs=["params:scenario_params", "parameters"],
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
        tags="build_runners",
    )
    return pipe
