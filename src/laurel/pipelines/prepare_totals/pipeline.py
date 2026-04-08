"""Kedro pipeline definition for the ``prepare_totals`` pipeline.

Wires the nodes from :mod:`.nodes` into a single ``Pipeline`` object.
For full documentation of each node's inputs, outputs, and algorithm,
see :mod:`.nodes`.

Sub-pipelines / tags
--------------------
- **vius_pipe** — filters and aggregates VIUS microdata into a
  disaggregation crosswalk ``P(region, op_dist | weight_class)``.
- **adopt_pipe** — aggregates NREL Ledna adoption forecasts to the
  required group level.
- **joint_pipe** — merges the VIUS crosswalk onto the adoption totals
  to produce stratum-level vehicle counts (``adoption_scenarios_no_mandates``).
- **mandate_pipe** — expands ACF mandate schedules and appends a
  mandate-adjusted copy of the adoption projections
  (``adoption_scenarios``).

To visualise the node graph interactively, run::

    kedro viz run

then open http://localhost:4141 in a browser and select ``prepare_totals``
from the pipeline dropdown.
"""

from kedro.pipeline import Node, Pipeline

from laurel.utils.data import filter_by_vals_in_cols
from laurel.utils.params import set_entity_params

from .nodes import (
    aggregate_adoption_forecast_totals,
    aggregate_vius_totals,
    build_mandates_by_group,
    concat_projections_with_mandates,
    create_disaggregated_adoption,
    prepare_for_merging,
)


def create_pipeline(**kwargs) -> Pipeline:
    vius_pipe = Pipeline(
        [
            Node(
                func=filter_by_vals_in_cols,
                inputs=["vius_public_use", "params:filter_vius_in_use"],
                outputs="vius_in_use",
                name="filter_vius_in_use",
            ),
            Node(
                func=prepare_for_merging,
                inputs=["vius_in_use", "params:prepare_for_merging"],
                outputs="vius_ready_to_merge",
                name="prepare_for_merging",
            ),
            Node(
                func=set_entity_params,
                inputs=["vius_ready_to_merge", "params:vius_classifications"],
                outputs="vius_classified",
                name="add_vius_classes",
            ),
            Node(
                func=aggregate_vius_totals,
                inputs=["vius_classified", "params:aggregate_vius_totals"],
                outputs="vius_aggregated",
                name="aggregate_vius_totals",
            ),
        ],
    )

    adopt_pipe = Pipeline(
        [
            Node(
                func=set_entity_params,
                inputs=["ledna_adoption", "params:adoption_forecast_classifications"],
                outputs="adoption_classified",
                name="add_adoption_classes",
            ),
            Node(
                func=aggregate_adoption_forecast_totals,
                inputs=[
                    "adoption_classified",
                    "params:aggregate_adoption_forecast_totals",
                ],
                outputs="adoption_aggregated",
                name="aggregate_adoption_forecast_totals",
            ),
        ],
    )

    joint_pipe = Pipeline(
        [
            Node(
                func=create_disaggregated_adoption,
                inputs=[
                    "adoption_aggregated",
                    "vius_aggregated",
                    "params:create_disaggregated_adoption",
                ],
                outputs="adoption_scenarios_no_mandates",
                name="create_disaggregated_adoption",
            ),
        ],
    )

    mandate_pipe = Pipeline(
        [
            Node(
                func=build_mandates_by_group,
                inputs=[
                    "advanced_clean_fleets_milestones",
                    "advanced_clean_trucks_states",
                    "params:build_mandates_by_group",
                ],
                outputs="mandate_projections",
                name="build_mandates_by_group",
            ),
            Node(
                func=concat_projections_with_mandates,
                inputs=[
                    "adoption_scenarios_no_mandates",
                    "mandate_projections",
                    "params:concat_projections_with_mandates",
                ],
                outputs="adoption_scenarios",
                name="concat_projections_with_mandates",
            ),
        ],
    )

    return vius_pipe + adopt_pipe + joint_pipe + mandate_pipe
