"""Kedro pipeline definition for the ``describe_vehicles`` pipeline.

Wires the nodes from :mod:`laurel.pipelines.describe_vehicles.nodes` into a single ``Pipeline`` object.
For full documentation of each node's inputs, outputs, and algorithm,
see :mod:`laurel.pipelines.describe_vehicles.nodes`.

Sub-pipelines / tags
--------------------
- **prep_spatial_dwells** — adds H3-centroid geometries to the ``DwellSet``
  and re-partitions it for on-disk checkpointing.
- **describe_vehs** — extracts vehicle-level attributes; computes
  time-weighted geographic centers, operating radii, observation frames,
  and primary operating-distance class for each vehicle.

To visualise the node graph interactively, run::

    kedro viz run

then open http://localhost:4141 in a browser and select ``describe_vehicles``
from the pipeline dropdown.
"""

from kedro.pipeline import Node, Pipeline

from laurel.models.dwell_sets import load_dwell_set, save_dwell_set

from .nodes import (
    filter_dwells_for_op_segment,
    get_operating_radius,
    get_operating_segment,
    get_time_weighted_centers,
    get_vehicle_observation_frames,
    mark_weight_class_group,
    partition_dwellset,
    spatialize_dwells,
    strip_vehicle_attrs,
)


def create_pipeline(**kwargs) -> Pipeline:
    geo_prep_pipe = Pipeline(
        [
            Node(
                func=load_dwell_set,
                inputs=["dwells_with_locations_dask", "params:load_dwell_set"],
                outputs="dwell_obj_desc_vehs",
                name="load_dwell_set_desc_vehs",
            ),
            Node(
                func=filter_dwells_for_op_segment,
                inputs="dwell_obj_desc_vehs",
                outputs="dwell_obj_filtered_desc_vehs",
                name="filter_dwells_for_op_segment",
            ),
            Node(
                func=spatialize_dwells,
                inputs="dwell_obj_filtered_desc_vehs",
                outputs="dwell_obj_spatial",
                name="spatialize_dwells",
            ),
            Node(
                func=partition_dwellset,
                inputs=["dwell_obj_spatial", "params:repartition_spatial_dwells"],
                outputs="dwell_obj_spatial_repartitioned",
                name="partition_spatial_dwells",
            ),
            Node(
                func=save_dwell_set,
                inputs="dwell_obj_spatial_repartitioned",
                outputs="dwells_with_locations_dask_spatial",
                name="save_dwell_set_spatial",
            ),
        ],
        tags="prep_spatial_dwells",
    )

    veh_pipe = Pipeline(
        [
            Node(
                func=strip_vehicle_attrs,
                inputs=["trips_formatted", "params:strip_vehicle_attrs"],
                outputs="vehicles_raw",
                name="strip_vehicle_attrs",
            ),
            Node(
                func=mark_weight_class_group,
                inputs=[
                    "vehicles_raw",
                    "params:weight_class_group",
                ],
                outputs="vehs_with_weight_class_group",
                name="mark_weight_class_group",
            ),
            Node(
                func=load_dwell_set,
                inputs=["dwells_with_locations_dask_spatial", "params:load_dwell_set"],
                outputs="dwell_obj_spatial_reread",
                name="load_dwell_set_spatial",
            ),
            Node(
                func=get_vehicle_observation_frames,
                inputs=[
                    "vehs_with_weight_class_group",
                    "dwell_obj_spatial_reread",
                    "params:observation_frames",
                ],
                outputs="vehicles_with_obs",
                name="get_vehicle_observation_frames",
            ),
            Node(
                func=get_operating_radius,
                inputs=[
                    "vehicles_with_obs",
                    "dwell_obj_spatial_reread",
                    "params:operating_radius",
                ],
                outputs="vehicles_with_radii",
                name="get_operating_radius",
            ),
            Node(
                func=get_time_weighted_centers,
                inputs=[
                    "vehicles_with_radii",
                    "dwell_obj_spatial_reread",
                    "params:time_weighted_centers",
                ],
                outputs="vehicles_with_centers",
                name="get_time_weighted_centers",
            ),
            Node(
                func=get_operating_segment,
                inputs=[
                    "vehicles_with_centers",
                    "dwell_obj_spatial_reread",
                    "params:operating_segment",
                ],
                outputs="vehicles_labelled",
                name="get_operating_segment",
            ),
        ],
        tags="describe_vehs",
    )

    return geo_prep_pipe + veh_pipe
