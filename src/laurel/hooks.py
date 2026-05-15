"""Kedro project hooks for the ``laurel`` package.

Currently provides :class:`DaskClusterHook`, which creates a
``dask.distributed.LocalCluster`` before a pipeline run begins so that all
catalog Dask-dataset reads and node computations share a single, correctly
sized cluster from the very start of execution.

Key design decisions
--------------------
- **before_pipeline_run**: The hook fires after the catalog is ready but
  before any node executes or any catalog dataset is loaded as a node input.
  This is the earliest point at which worker memory limits can be enforced
  for every Dask operation in the run.
- **Naming convention**: The hook looks for a params key named
  ``dask_{pipeline_name}`` (e.g. ``dask_evaluate_impacts`` for the
  ``evaluate_impacts`` pipeline).  No explicit mapping is required — adding
  cluster configuration for a new pipeline just means adding the correctly
  named key to its parameters file.  A warning is emitted if no such key
  exists, so mis-spellings and missing configs are visible at run time.
- **verbose toggle**: If ``verbose: false`` (the default), the hook raises the
  main-process ``distributed`` logger to ``WARNING`` and passes
  ``silence_logs=logging.WARNING`` to ``LocalCluster`` to quiet worker
  sub-processes.  The saved level is restored in ``_shutdown()`` after both
  closes complete.  Set ``verbose: true`` in the ``dask_*`` params block to
  re-enable full Dask logging and the hook's own ``INFO`` startup message.
- **Idempotent shutdown**: ``_shutdown()`` nulls out ``_cluster``,
  ``_client``, and ``_distributed_log_level`` after the first close, so
  calling it from both ``after_pipeline_run`` and ``on_pipeline_error``
  (which Kedro may do on error) is always safe.
- **use_dask toggle**: If the resolved params dict contains
  ``use_dask: false``, the hook skips cluster creation.  Node functions that
  previously accepted the client as a DAG-ordering sentinel now simply omit
  it; functions that call the client explicitly (e.g.
  :func:`~laurel.pipelines.evaluate_impacts.nodes.sample_profiles_node`)
  resolve the client via ``distributed.get_client()`` and handle the
  ``ValueError`` raised when no cluster is running.
"""

from __future__ import annotations

import logging
from typing import Any

from dask.distributed import Client, LocalCluster
from kedro.framework.hooks import hook_impl
from kedro.io.core import DatasetNotFoundError

logger = logging.getLogger(__name__)


class DaskClusterHook:
    """Owns the Dask ``LocalCluster`` lifecycle for a full pipeline run.

    Creates the cluster in ``before_pipeline_run`` so that all catalog Dask
    dataset reads and node computations share one correctly sized cluster with
    the configured memory limits.  Tears it down in ``after_pipeline_run`` and
    ``on_pipeline_error``.

    Cluster configuration is read from a params key named
    ``dask_{pipeline_name}`` by convention; see the module docstring for
    details.
    """

    def __init__(self) -> None:
        self._cluster: LocalCluster | None = None
        self._client: Client | None = None
        self._distributed_log_level: int | None = None

    @hook_impl
    def before_pipeline_run(
        self, run_params: dict[str, Any], pipeline, catalog
    ) -> None:
        """Start a ``LocalCluster`` sized according to the pipeline's Dask params."""
        pipeline_name = run_params.get("pipeline_name", "")
        if not pipeline_name:
            return
        param_key = f"params:dask_{pipeline_name}"
        try:
            params = catalog.load(param_key)
        except DatasetNotFoundError:
            logger.warning(
                "DaskClusterHook: no Dask configuration found for pipeline '%s' "
                "(expected params key 'dask_%s'). Running without a managed cluster.",
                pipeline_name,
                pipeline_name,
            )
            return
        if not params.get("use_dask", True):
            return
        verbose = params.get("verbose", False)
        cluster_kwargs = params.get("cluster") or {}
        if not verbose:
            dist_logger = logging.getLogger("distributed")
            self._distributed_log_level = dist_logger.level
            dist_logger.setLevel(logging.WARNING)
            cluster_kwargs = {"silence_logs": logging.WARNING, **cluster_kwargs}
        self._cluster = LocalCluster(**cluster_kwargs)
        logger.info(
            "DaskClusterHook: started LocalCluster for pipeline '%s' (%s).",
            pipeline_name,
            ", ".join(f"{k}={v}" for k, v in cluster_kwargs.items())
            if cluster_kwargs
            else "default settings",
        )
        self._client = Client(self._cluster)

    @hook_impl
    def after_pipeline_run(
        self, run_params: dict[str, Any], run_result: dict[str, Any], pipeline, catalog
    ) -> None:
        """Shut down the cluster after a successful pipeline run."""
        self._shutdown()

    @hook_impl
    def on_pipeline_error(
        self, error: Exception, run_params: dict[str, Any], pipeline, catalog
    ) -> None:
        """Shut down the cluster if the pipeline raises an uncaught exception."""
        self._shutdown()

    def _shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._cluster is not None:
            self._cluster.close()
            self._cluster = None
        if self._distributed_log_level is not None:
            logging.getLogger("distributed").setLevel(self._distributed_log_level)
            self._distributed_log_level = None
