"""
This is a boilerplate pipeline 'build_runners'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Node, Pipeline

from megaplug.scenarios.cmd import generate_bash_script

from .nodes import generate_scenario_configs


def create_pipeline(**kwargs) -> Pipeline:
    pipe = Pipeline(
        [
            Node(
                func=generate_scenario_configs,
                inputs=["params:scenario_params", "parameters"],
                outputs=["scenario_configs", "scenario_builder"],
                name="generate_scenario_configs",
            ),
            Node(
                func=generate_bash_script,
                inputs=[
                    "params:slurm_command",
                    "scenario_builder",
                    "params:cmd_line_calls",
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
