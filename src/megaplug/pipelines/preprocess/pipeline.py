"""Kedro pipeline definition for the ``preprocess`` pipeline.

Wires the nodes from :mod:`.nodes` into a single ``Pipeline`` object.
For full documentation of each node's inputs, outputs, and algorithm,
see :mod:`.nodes`.

Sub-pipelines / tags
--------------------
- **vius_scaling** — builds survey-weight-adjusted HDT fleet totals from
  VIUS microdata for use as scaling denominators.

To visualise the node graph interactively, run::

    kedro viz run

then open http://localhost:4141 in a browser and select ``preprocess``
from the pipeline dropdown.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    build_vius_scaling_totals,
)


# TODO: Consider removing this pipeline, or shifting pieces of it to prepare_totals
def create_pipeline(**kwargs) -> Pipeline:
    scale_pipe = Pipeline(
        [
            Node(
                func=build_vius_scaling_totals,
                inputs=[
                    "vius_public_use",
                    "params:build_vius_scaling_totals",
                ],
                outputs="vius_scaling",
                name="build_vius_scaling_totals",
            ),
        ],
        tags="vius_scaling",
    )

    return scale_pipe
