"""Helpers for managing Dask LocalCluster lifecycle and materialising deferred data.

Kedro pipelines that process large DwellSets or partitioned datasets use Dask
for parallelism.  This module provides three thin wrappers — one to start a
``LocalCluster``, one to shut it down cleanly, and one to force a Dask
DataFrame (or a DwellSet backed by one) into RAM — so that the cluster
lifecycle appears as ordinary Kedro nodes in the pipeline graph.

Key design decisions
--------------------
- **Soft Dask toggle**: If ``params["use_dask"]`` is ``False``, :func:`start_dask_node`
  returns the string sentinel ``"None"`` (not Python ``None``) so that downstream
  nodes that receive the client still have a truthy value to pass through the
  pipeline graph without triggering Kedro catalog mismatches.
- **Auto cluster**: If ``params["cluster"]`` is ``None``, :func:`start_dask_node`
  calls ``Client()`` with no arguments so Dask selects cluster parameters
  automatically from local resources.  The auto-created cluster is returned via
  ``client.cluster`` so that :func:`stop_dask_node` can shut it down cleanly.
- **result dependency**: :func:`stop_dask_node` accepts the final computed
  dataset as a ``result`` argument purely to enforce DAG ordering; Kedro
  executes nodes only once all their inputs are ready, so passing the last
  dataset here guarantees the cluster outlives all computation.
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster

from laurel.models.dwell_sets import DwellSet


def start_dask_node(params: dict) -> tuple[LocalCluster, Client]:
    """Start a Dask ``LocalCluster`` and connect a ``Client`` to it.

    If ``params["use_dask"]`` is explicitly set to ``False``, returns the string
    sentinel ``("None", "None")`` instead of a real cluster/client pair so that
    downstream nodes can be written uniformly without ``None``-checks.

    If ``params["cluster"]`` is ``None``, calls ``Client()`` with no arguments
    so that Dask creates a ``LocalCluster`` automatically using all available
    local resources.  The auto-created cluster is accessible via
    ``client.cluster`` and is returned as the first element of the tuple so
    that :func:`stop_dask_node` can shut it down cleanly.

    Args:
        params: Configuration dict with the following keys:

            - **use_dask** (``bool``, optional): If ``False``, skip cluster
              creation.  Defaults to ``True`` when absent.
            - **cluster** (``dict`` or ``None``): Keyword arguments forwarded to
              ``dask.distributed.LocalCluster`` (e.g. ``n_workers``,
              ``threads_per_worker``, ``memory_limit``).  If ``None``, Dask
              selects sensible defaults for the local machine automatically.

    Returns:
        A ``(LocalCluster, Client)`` pair, or ``("None", "None")`` if Dask is
        disabled.
    """
    if ("use_dask" in params and params["use_dask"]) or ("use_dask" not in params):
        if params.get("cluster") is None:
            client = Client()
            return client.cluster, client
        else:
            cluster = LocalCluster(**params["cluster"])
            client = Client(cluster)
            return cluster, client
    else:
        return "None", "None"


def stop_dask_node(cluster: LocalCluster, client: Client, result: object) -> None:
    """Shut down the Dask ``Client`` and ``LocalCluster``.

    The ``result`` parameter serves only as a DAG dependency: by wiring the
    final computed dataset through this node, Kedro guarantees the cluster
    remains alive until all upstream computation has finished.

    Handles the ``"None"`` string sentinel returned by :func:`start_dask_node`
    when Dask is disabled, so this node is always safe to include in the
    pipeline.

    Args:
        cluster: The ``LocalCluster`` to close, or the string ``"None"`` if Dask
            was disabled.
        client: The ``Client`` to close, or the string ``"None"`` if Dask was
            disabled.
        result: The final dataset produced by the Dask computation.  Not used
            directly; present only to enforce execution ordering.
    """
    if client != "None" and isinstance(client, Client):
        client.close()

    if cluster != "None" and isinstance(cluster, LocalCluster):
        cluster.close()


def load_in_memory_node(ddf: dd.DataFrame | DwellSet) -> pd.DataFrame | DwellSet:
    """Force a Dask DataFrame (or DwellSet) into in-memory pandas form.

    Used as a Kedro node to materialise a deferred Dask computation before
    operations that require random access or pandas-only APIs (e.g. Numba JIT
    calls, index-based joins).  If the input is already a pandas DataFrame or
    a DwellSet backed by one, it is returned unchanged.

    Args:
        ddf: A Dask ``DataFrame`` or a ``DwellSet`` whose ``data`` attribute may
            be a Dask ``DataFrame``.

    Returns:
        A pandas ``DataFrame``, or a ``DwellSet`` whose ``data`` attribute is a
        pandas ``DataFrame``.

    Raises:
        NotImplementedError: If ``ddf`` is neither a Dask/pandas DataFrame nor a
            ``DwellSet``.
    """
    if isinstance(ddf, DwellSet):
        if isinstance(ddf.data, dd.DataFrame):
            ddf_new = ddf.copy_without_data()
            ddf_new.data = ddf.data.compute()
            return ddf_new
        else:
            return ddf
    elif isinstance(ddf, dd.DataFrame):
        return ddf.compute()
    else:
        raise NotImplementedError(
            "Load-in-memory is not yet implemented for this type."
        )
