"""Nodes for the ``download_inputs`` pipeline.

These nodes only *trigger* downloads. Both endpoints of each transfer -- the source URL
and the destination path -- are configured on the catalog entry the node writes to, so
moving to a new source is a catalog edit rather than a code or parameters change. The
transfer itself belongs to
:class:`laurel.datasets.downloaded_file.DownloadedFileDataset`.
"""

import logging

logger = logging.getLogger(__name__)


def request_osm_download() -> dict:
    """Triggers the OpenStreetMap download configured in the catalog.

    Returns:
        Overrides layered over the ``osm_north_america`` entry's ``save_args`` for this
        run. Empty by default: the source URL, the destination and the transfer settings
        all come from the catalog. Returning ``{"overwrite": True}`` would force a
        re-fetch of a file that is already present.
    """
    logger.info("Requesting the OpenStreetMap download configured in the catalog.")
    return {}
