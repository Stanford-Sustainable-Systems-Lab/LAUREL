"""Abstract base classes for Kedro scenario configuration generation.

Provides :class:`ScenarioBuilder`, the abstract interface that concrete scenario
builders must implement to generate a Kedro configuration file tree for a
batch of scenario runs (States of the World).  Each concrete builder maps a
parameter grid to a directory layout that is compatible with SLURM job arrays
and Kedro's ``PartitionedDataset`` format.

The generated layout follows the pattern::

    <scenario_name>/<intermediate_dirs>/task_<N>/parameters.yml

where ``N`` is the zero-based SLURM array task index.  At runtime each task
discovers its own config directory, selects the matching parameter file, and
writes its output under the same ``task_<N>`` subtree.

Key design decisions
--------------------
- **``task_N`` directory layout**: giving each task its own subdirectory
  avoids filesystem conflicts when SLURM array jobs run concurrently, and
  makes it trivial for :class:`~megaplug.scenarios.read.ScenarioReader` to
  filter completed vs. pending tasks by path prefix.
- **``results_partition`` injection**: the generated config dict always
  contains a ``"results_partition"`` key with the task directory and
  ``partition_level_names``.  The pipeline nodes ``write_scenario_partition``
  and ``read_scenario_partition`` in :mod:`megaplug.scenarios.io` read this
  key at runtime to direct I/O to the correct location without hardcoding
  paths.
- **``n_tasks_generated`` side-effect**: :meth:`ScenarioBuilder.build_configs`
  sets this attribute so that :class:`~megaplug.scenarios.cmd.ScenarioBashWriter`
  can generate the correct ``--array=0-N`` SLURM range without a separate
  counting step.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np


class ScenarioBuilder(ABC):
    """Abstract base class for scenario configuration builders.

    Concrete subclasses define how a parameter grid maps to Kedro configuration
    partitions.  The framework uses the ``SLURM_ARRAY_TASK_ID`` environment
    variable to select the correct partition at runtime, or a local ``for``
    loop for serial execution.

    Class attributes:
        scen_param_key: Key under which the partition directory and level names
            are injected into each generated config dict.  Defaults to
            ``"results_partition"``.
        n_tasks_generated: Set to the total number of tasks by
            :meth:`build_configs`; initially ``None``.
    """

    scen_param_key: str = "results_partition"
    n_tasks_generated: int | None = None

    def __init__(self, scen_params: dict, all_params: dict) -> None:
        """Initialise a scenario builder.

        Args:
            scen_params: Scenario-specific parameters dict.  Must contain the
                key ``"display_name"`` (a human-readable scenario identifier
                used as the root directory name).  May contain additional keys
                consumed by :meth:`_build_param_dicts` in concrete subclasses
                (e.g. ``"builder"``, ``"n_scenarios"``, ``"seed"``).
            all_params: The full Kedro ``parameters`` dict for the run,
                forwarded to concrete builders that need pipeline-level
                parameters such as base paths or default energy rates.

        Raises:
            RuntimeError: If ``"display_name"`` is absent from ``scen_params``.
        """
        self.scen_params = scen_params
        if "display_name" in self.scen_params:
            self.display_name = self.scen_params["display_name"]
        else:
            raise RuntimeError(
                "The 'display_name' for the scenario was not in the scenario parameters."
            )
        self.params = all_params

    @property
    @abstractmethod
    def partition_level_names(self) -> tuple[str]:
        """Ordered tuple of path-component names that describe each partition level.

        These names correspond to the path components between the scenario root
        and the ``task_N`` leaf directory (e.g. ``("run_name", "task_id")``).
        :class:`~megaplug.scenarios.read.ScenarioReader` uses them to extract
        structured metadata from partition paths.
        """
        pass  # Implement this in the concrete classes by setting the attribute partition_level_names

    @abstractmethod
    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        """Build the (filepath, parameter dict) pairs that define this scenario set.

        Returns:
            A two-tuple ``(paths, param_dicts)`` of equal-length lists.  Each
            ``paths[i]`` is the base directory for task ``i``; each
            ``param_dicts[i]`` is the Kedro parameter dict for that task
            (excluding the ``results_partition`` key, which is added by
            :meth:`_build_single_partition`).
        """
        pass

    def _build_single_partition(
        self: Self,
        pth: Path,
        params: dict,
        task_id: int,
    ) -> dict[str, dict]:
        """Build a Kedro configuration partition dict for a single task.

        Appends a ``task_<task_id>`` leaf to ``pth``, injects the
        ``results_partition`` key into ``params``, and returns a dict mapping
        the config file path to the completed parameter dict.

        Args:
            pth: Base directory for this task's output, down to (but not
                including) the ``task_N`` leaf.
            params: Parameter dict for this task.  Modified in-place to add
                the ``results_partition`` entry.
            task_id: Zero-based task index (matches ``SLURM_ARRAY_TASK_ID``).

        Returns:
            Single-entry dict ``{config_file_path: params}`` ready for merging
            into the full partitions dict returned by :meth:`build_configs`.
        """
        part_pth = pth / f"task_{task_id}"
        # Add the partition path to this scenario's config
        result_partition = {
            "dir": str(part_pth),
            "level_names": list(self.partition_level_names),
        }
        params.update({self.scen_param_key: result_partition})

        # Add this scenario's config to the partitions
        param_file = str(part_pth / "parameters")
        return {param_file: params}

    def build_configs(self: Self) -> dict[Path, dict]:
        """Generate the full set of Kedro configuration partition dicts.

        Calls :meth:`_build_param_dicts`, then wraps each ``(path, params)``
        pair with :meth:`_build_single_partition` to inject the
        ``results_partition`` key.  Sets :attr:`n_tasks_generated` as a
        side-effect so the bash writer can generate the correct SLURM array
        range.

        Returns:
            Dict mapping each task's config file path to its parameter dict.
            Suitable for saving as a Kedro ``PartitionedDataset``.
        """
        paths, scen_params = self._build_param_dicts()
        parts = {}
        for task, (path, scen) in enumerate(zip(paths, scen_params)):
            cur_part = self._build_single_partition(pth=path, params=scen, task_id=task)
            parts.update(cur_part)
        self.n_tasks_generated = len(parts)
        return parts

    @staticmethod
    def _get_random_seed(seed: int, max: int = 1000) -> int:
        """Derive a bounded random integer from a base seed.

        Uses ``numpy.random.default_rng`` so that the output is reproducible
        given the same ``seed`` while avoiding simple modular patterns.

        Args:
            seed: Base integer seed for the RNG.
            max: Exclusive upper bound for the output integer.  Defaults to
                ``1000``.

        Returns:
            A random integer in ``[0, max)``.
        """
        rng = np.random.default_rng(seed=seed)
        num = rng.integers(max)
        return int(num)


class TestScenarioBuilder(ScenarioBuilder):
    """Minimal concrete builder that produces a single-task scenario partition.

    Used in tests and as a reference implementation for :class:`ScenarioBuilder`.
    Generates one partition whose output directory is ``<display_name>/task_0/``
    with no extra parameters.
    """

    partition_level_names = ("run_name", "task_id")

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        """Return a single (path, empty-params) pair for the test scenario.

        Returns:
            Tuple of ``([Path(display_name)], [{}])``.
        """
        paths = [Path(self.display_name)]
        scens = [{}]
        return (paths, scens)
