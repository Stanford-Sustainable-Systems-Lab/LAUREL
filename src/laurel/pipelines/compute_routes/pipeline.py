"""Kedro pipeline definition for the ``compute_routes`` pipeline.

Wires the nodes from :mod:`laurel.pipelines.compute_routes.nodes` into a single ``Pipeline`` object.
For full documentation of each node's inputs, outputs, and algorithm,
see :mod:`laurel.pipelines.compute_routes.nodes`.

Sub-pipelines / tags
--------------------
- **import** — imports the OSM road network into the GraphHopper Docker
  container (run once; reuses the pre-built graph on subsequent runs).
- **pre_routing** — filters short trips, converts H3 hex origins/destinations
  to point geometries, and re-partitions for checkpointing.
- **routing** — starts a Dask cluster and the GraphHopper server, computes
  shortest-path routes for all trips, then tears both down.
- **insert_optional_stops** — spatially joins truck-stop candidates onto
  routes, projects stop positions along each route, splits trips at stops,
  and concatenates the result with the original trips.

To visualise the node graph interactively, run::

    kedro viz run

then open http://localhost:4141 in a browser and select ``compute_routes``
from the pipeline dropdown.
"""

from kedro.pipeline import Node, Pipeline

from laurel.routing.nodes import (
    start_routing_server_node,
    stop_routing_server_node,
)
from laurel.utils.data import filter_by_vals_in_cols
from laurel.utils.distributed import start_dask_node, stop_dask_node

from .nodes import (
    concat_optional_stops,
    describe_optional_stop_trips,
    filter_routable_trips,
    format_stop_locations,
    get_optional_stop_trips,
    get_routes_node,
    get_trip_orig_dest_points,
    import_graph,
    partition_trips,
)


def create_pipeline(**kwargs) -> Pipeline:
    import_pipe = Pipeline(
        [
            Node(
                func=import_graph,
                inputs="params:graphhopper",
                outputs=None,
                name="import_graph",
            ),
        ],
        tags="import",
    )

    pre_route_pipe = Pipeline(
        [
            Node(
                func=filter_routable_trips,
                inputs=["trips_formatted", "params:filter_routable_trips"],
                outputs="trips_routable_filtered",
                name="filter_routable_trips",
            ),
            Node(
                func=get_trip_orig_dest_points,
                inputs=["trips_routable_filtered", "params:get_trip_orig_dest_points"],
                outputs="trips_to_route_big_partitions",
                name="get_trip_orig_dest_points",
            ),
            Node(
                func=partition_trips,
                inputs=["trips_to_route_big_partitions", "params:partition_trips"],
                outputs="trips_to_route",
                name="partition_trips",
            ),
        ],
        tags="pre_routing",
    )

    route_pipe = Pipeline(
        [
            Node(
                func=start_dask_node,
                inputs="params:dask_routing",
                outputs=["dask_cluster_routing", "dask_client_routing"],
                name="start_dask_routing",
            ),
            Node(
                func=start_routing_server_node,
                inputs=[
                    "params:graphhopper",
                ],
                outputs="routing_server",
                name="start_routing_server",
            ),
            Node(
                func=get_routes_node,
                inputs=[
                    "trips_to_route",
                    "routing_server",
                    "params:get_routes",
                ],
                outputs="trips_routed",
                name="get_routes",
            ),
            Node(
                func=stop_routing_server_node,
                inputs=["routing_server", "trips_routed"],
                outputs=None,
                name="stop_routing_server",
            ),
            Node(
                func=stop_dask_node,
                inputs=["dask_cluster_routing", "dask_client_routing", "trips_routed"],
                outputs=None,
                name="stop_dask_routing",
            ),
        ],
        tags="routing",
    )

    opt_stops_pipe = Pipeline(
        [
            Node(
                func=filter_by_vals_in_cols,
                inputs=["hex_cluster_corresp", "params:filter_opt_stops"],
                outputs="parking_filtered",
                name="filter_truck_stops",
            ),
            Node(
                func=format_stop_locations,
                inputs=["parking_filtered", "params:format_opt_stop_locations"],
                outputs="parking_formatted",
                name="format_stop_locations",
            ),
            Node(
                func=get_optional_stop_trips,
                inputs=[
                    "trips_routed",  # Use the `compute_routes` pipeline to get this
                    "parking_formatted",
                    "params:get_optional_stop_trips",
                ],
                outputs="optional_stop_trips_raw",
                name="get_optional_stop_trips",
            ),
            Node(
                func=describe_optional_stop_trips,
                inputs=[
                    "optional_stop_trips_raw",
                    "params:describe_optional_stop_trips",
                ],
                outputs="optional_stop_trips",
                name="describe_optional_stop_trips",
            ),
            Node(
                func=concat_optional_stops,
                inputs=[
                    "trips_formatted",
                    "optional_stop_trips",
                    "params:concat_optional_stops",
                ],
                outputs="trips_with_optional",
                name="concat_optional_stops",
            ),
        ],
        tags="insert_optional_stops",
    )

    return import_pipe + pre_route_pipe + route_pipe + opt_stops_pipe
