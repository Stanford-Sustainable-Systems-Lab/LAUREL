"""Kedro pipeline definition for the ``build_runners`` pipeline.

Wires the nodes from :mod:`.nodes` into a single ``Pipeline`` object.
For full documentation of each node's inputs, outputs, and algorithm,
see :mod:`.nodes`.

Sub-pipelines / tags
--------------------
- **build_runners** — generates per-scenario YAML configuration
  partitions and the corresponding SLURM batch script for HPC execution.

To visualise the node graph interactively, run::

    kedro viz run

then open http://localhost:4141 in a browser and select ``build_runners``
from the pipeline dropdown.
"""

from kedro.pipeline import Node, Pipeline

from laurel.scenarios.cmd import generate_bash_script

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
