from kedro.pipeline import Node, Pipeline  # noqa

from megaPLuG.models.dwell_sets import save_dwell_set

from .nodes import coalesce_interrupted_dwells, create_dwells, calc_rolling_dwell_ratios


def create_pipeline(**kwargs) -> Pipeline:
    dwell_pipe = Pipeline(
        [
            # If you want optional stops, then use "trips_with_optional" as the input.
            # Otherwise, use "trips_formatted". Also, if you want optional stops, run
            # the `optional_stops` and this `create_dwells` pipeline together, using the
            # "create_dwells_optional_stops" tag for convenience.
            Node(
                func=create_dwells,
                inputs=["trips_with_optional", "params:create_dwells"],
                outputs="dwell_obj_preprocess",
                name="create_dwells",
            ),
            Node(
                func=coalesce_interrupted_dwells,
                inputs=["dwell_obj_preprocess", "params:coalesce_interrupted_dwells"],
                outputs="dwell_obj_coalesced",
                name="coalesce_interrupted_dwells",
            ),
            Node(
                func=calc_rolling_dwell_ratios,
                inputs=[
                    "dwell_obj_coalesced",
                    "params:rolling_dwell_ratios",
                ],
                outputs="dwell_obj_roll",
                name="calc_rolling_dwell_ratios",
            ),
            # TODO: Add a node here which maps on the clusters by hex_id
            Node(
                func=save_dwell_set,
                inputs="dwell_obj_roll",
                outputs="dwells",
                name="save_dwell_set_preprocess",
            ),
        ],
        tags=["create_dwells", "create_dwells_optional_stops"],
    )

    return dwell_pipe
