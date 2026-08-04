"""Kedro pipeline definition for the ``download_inputs`` pipeline.

Wires the nodes from :mod:`laurel.pipelines.download_inputs.nodes` into a single
``Pipeline`` object. For full documentation of each node's inputs, outputs, and
algorithm, see :mod:`laurel.pipelines.download_inputs.nodes`.

This pipeline acquires large public raw inputs that other pipelines consume. It is
separate from those pipelines because a single download can serve several of them: the
OpenStreetMap extract feeds both ``compute_routes`` (as Graphhopper's road network) and
``describe_locations`` (as a source of truck stops and warehouses).

What gets downloaded from where is configured entirely on the catalog entries these nodes
write to, in ``conf/base/catalog.yml``. Downloads are skipped when the target file is
already present, so re-running this pipeline is cheap; see
:class:`laurel.datasets.downloaded_file.DownloadedFileDataset`.

Sub-pipelines / tags
--------------------
- **osm** — downloads the OpenStreetMap PBF extract for North America.

To visualise the node graph interactively, run::

    kedro viz run

then open http://localhost:4141 in a browser and select ``download_inputs``
from the pipeline dropdown.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import request_osm_download


def create_pipeline(**kwargs) -> Pipeline:
    osm_pipe = Pipeline(
        [
            Node(
                func=request_osm_download,
                inputs=None,
                outputs="osm_north_america",
                name="download_osm",
            ),
        ],
        tags="osm",
    )
    return osm_pipe
