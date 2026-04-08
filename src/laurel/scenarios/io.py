"""Kedro pipeline node functions for reading and writing scenario-partitioned datasets.

Provides two thin wrapper nodes that bridge the generic Kedro
``PartitionedDataset`` API and the scenario directory layout defined by
:class:`~laurel.scenarios.build.ScenarioBuilder`:

- :func:`write_scenario_partition` — packages an arbitrary object for Kedro to
  save under the current task's output directory.
- :func:`read_scenario_partition` — loads a single partition from that
  directory, enabling within-pipeline re-reads of just-written data.

Key design decisions
--------------------
- **Node functions, not methods**: these functions are inserted directly into a
  Kedro pipeline graph as nodes.  The partition path comes from the runtime
  ``results_partition`` parameter injected into each task's config by
  :meth:`~laurel.scenarios.build.ScenarioBuilder._build_single_partition`,
  so the nodes need no knowledge of the scenario structure themselves.
- **Single-partition reads only**: :func:`read_scenario_partition` is designed
  for the common case of reading back the one partition written by the current
  task (e.g. to pass its output to a downstream node in the same run).  For
  multi-scenario aggregation across all States of the World, use
  :meth:`~laurel.scenarios.read.ScenarioReader.read_partitions` instead.
"""

from pathlib import Path

from dask.distributed import Client

from .read import ScenarioReader


def write_scenario_partition(obj: object, params: dict) -> dict[str, object]:
    """Package an object for Kedro to save to the current scenario's output directory.

    Wraps ``obj`` in a single-entry partition dict keyed by the task directory
    path.  Kedro's ``PartitionedDataset`` machinery uses this dict to determine
    where and how to serialise the object (format is controlled by
    ``catalog.yml``).

    Args:
        obj: The dataset to save (e.g. a ``pd.DataFrame``, ``dict``, or any
            object supported by the catalog entry).
        params: The ``"results_partition"`` sub-dict from the task's Kedro
            parameter config.  Must contain:

            - ``"dir"`` *(str)*: the task output directory path.
            - ``"level_names"`` *(list[str])*: partition level name metadata
              (not used by this function but present in the dict).

    Returns:
        Single-entry dict ``{params["dir"]: obj}`` ready for Kedro to persist.
    """
    return {params["dir"]: obj}


def read_scenario_partition(
    partitions: dict, params: dict, client: Client = None
) -> object:
    """Load a single partition from the current scenario's output directory.

    Filters the full ``PartitionedDataset`` dict to the one entry whose path
    matches ``params["dir"]``, calls its loader function, and returns the
    result.  Raises if zero or more than one matching partition is found.

    For reading multiple partitions across scenarios, use
    :meth:`~laurel.scenarios.read.ScenarioReader.read_partitions` instead.

    Args:
        partitions: Full Kedro ``PartitionedDataset`` dict mapping partition
            path strings to zero-argument loader callables.
        params: The ``"results_partition"`` sub-dict from the task's Kedro
            parameter config.  Must contain ``"dir"`` *(str)* — the path of
            the partition to load.
        client: Unused Dask ``Client`` argument retained for pipeline
            compatibility (ensures Dask is started before this node runs).

    Returns:
        The loaded dataset object returned by the partition's loader callable.

    Raises:
        RuntimeError: If more than one partition matches ``params["dir"]``.
    """
    dir = [Path(params["dir"])]
    selected = ScenarioReader.select_partitions_static(partitions=partitions, dirs=dir)
    if len(selected) > 1:
        raise RuntimeError("More than one partition identified in the given directory.")
    part_load_func = list(selected.values())[0]
    data = part_load_func()
    return data
