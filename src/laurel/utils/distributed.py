"""Helpers for materialising deferred Dask data as a Kedro node.

Kedro pipelines that process large DwellSets or partitioned datasets use Dask
for parallelism.  Cluster lifecycle is now managed by
:class:`~laurel.hooks.DaskClusterHook`, which starts and stops a
``LocalCluster`` around every pipeline run.  This module exposes only the
:func:`load_in_memory_node` helper, which forces a Dask DataFrame (or a
DwellSet backed by one) into RAM as a regular Kedro node.

Key design decisions
--------------------
- **load_in_memory_node**: Used to materialise deferred Dask computations
  before operations that require random access or pandas-only APIs (e.g.
  Numba JIT calls, index-based joins).  Returns the input unchanged if it is
  already backed by pandas.
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd

from laurel.models.dwell_sets import DwellSet


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
