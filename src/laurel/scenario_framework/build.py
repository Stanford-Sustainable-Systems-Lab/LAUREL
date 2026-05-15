"""Abstract base classes for Kedro scenario configuration generation.

Provides :class:`ScenarioBuilder`, the abstract interface that concrete scenario
builders must implement to generate a Kedro configuration file tree for a
batch of scenario runs (States of the World).  Each concrete builder maps a
parameter grid to a directory layout that is compatible with SLURM job arrays
and Kedro's ``PartitionedDataset`` format.

The generated layout follows the pattern::

    <scenario_name>/<intermediate_dirs>/task_<N>/parameters.yml
    <scenario_name>/<intermediate_dirs>/task_<N>/catalog.yml

where ``N`` is the zero-based SLURM array task index.  The per-task
``catalog.yml`` overrides selected ``PartitionedDataset`` entries from
``conf/base/catalog.yml`` with direct single-file or single-directory entries
that point at the task-specific path.  This lets Kedro determine whether an
individual task has already run (via its standard "skip if output exists"
logic) without touching the base catalog.

Which entries are overridden is declared in ``conf/base/catalog.yml`` itself,
using the ``metadata.scenario_override`` field on each relevant entry.  The
base catalog therefore remains the single source of truth: adding a new
scenario-specific dataset only requires annotating it there.

Key design decisions
--------------------
- **``task_N`` directory layout**: giving each task its own subdirectory
  avoids filesystem conflicts when SLURM array jobs run concurrently, and
  makes it trivial for :class:`~laurel.scenario_framework.read.ScenarioReader` to
  filter completed vs. pending tasks by path prefix.
- **Per-task catalog overrides**: generated alongside each ``parameters.yml``
  so Kedro can check individual task completion.  Entries with
  ``flatten: true`` collapse a ``PartitionedDataset`` into the inner dataset
  type (e.g. ``pandas.FeatherDataset``) pointing at the task file; entries
  without ``flatten`` redirect ``path`` to the task subdirectory (used for
  debug partitions whose nodes return dicts).
- **``n_tasks_generated`` side-effect**: :meth:`ScenarioBuilder.build_configs`
  sets this attribute so that :class:`~laurel.scenario_framework.cmd.ScenarioBashWriter`
  can generate the correct ``--array=0-N`` SLURM range without a separate
  counting step.
"""

from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import ClassVar, Self

import numpy as np

_FILEPATH_TYPES: frozenset[str] = frozenset(
    {
        "pandas.FeatherDataset",
        "pandas.CSVDataset",
        "pandas.ParquetDataset",
        "pandas.ExcelDataset",
        "dask.ParquetDataset",
    }
)


class ScenarioBuilder(ABC):
    """Abstract base class for scenario configuration builders.

    Concrete subclasses define how a parameter grid maps to Kedro configuration
    partitions.  The framework uses the ``SLURM_ARRAY_TASK_ID`` environment
    variable to select the correct partition at runtime, or a local ``for``
    loop for serial execution.

    Class attributes:
        _registry: Maps each concrete subclass name to the class itself.
            Populated automatically by :meth:`__init_subclass__` when a
            subclass is defined (i.e. when its module is imported).
        n_tasks_generated: Set to the total number of tasks by
            :meth:`build_configs`; initially ``None``.
    """

    _registry: ClassVar[dict[str, type]] = {}
    n_tasks_generated: int | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register every concrete subclass by name on first import."""
        super().__init_subclass__(**kwargs)
        ScenarioBuilder._registry[cls.__name__] = cls

    def __init__(
        self,
        scen_params: dict,
        all_params: dict,
        catalog: dict,
    ) -> None:
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
            catalog: The raw ``conf/base/catalog.yml`` dict, loaded as
                plain YAML (no OmegaConf interpolation).
                :meth:`_build_single_partition` emits a ``catalog.yml``
                alongside each ``parameters.yml`` by calling
                :meth:`_build_single_catalog`.

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
        self.catalog = catalog

    @property
    @abstractmethod
    def partition_level_names(self) -> tuple[str]:
        """Ordered tuple of path-component names that describe each partition level.

        These names correspond to the path components between the scenario root
        and the ``task_N`` leaf directory (e.g. ``("run_name", "task_id")``).
        :class:`~laurel.scenario_framework.read.ScenarioReader` uses them to extract
        structured metadata from partition paths.
        """
        pass  # Implement this in the concrete classes by setting the attribute partition_level_names

    @abstractmethod
    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        """Build the (filepath, parameter dict) pairs that define this scenario set.

        Returns:
            A two-tuple ``(paths, param_dicts)`` of equal-length lists.  Each
            ``paths[i]`` is the base directory for task ``i``; each
            ``param_dicts[i]`` is the Kedro parameter dict for that task.
        """
        pass

    def _build_single_catalog(self, task_pth: Path) -> dict:
        """Generate a catalog override dict for a single task.

        Scans ``self.catalog`` for entries with ``metadata.scenario_override``
        and generates a per-task override for each.  Entries with ``flatten: true``
        collapse a PartitionedDataset into the inner dataset type (e.g.
        ``pandas.FeatherDataset``) pointing at the task-specific file.  Entries
        without ``flatten`` redirect ``path`` to the task subdirectory while
        preserving the dataset structure (used for debug partitions whose nodes
        return dicts).

        The per-task path is always ``base_path / task_pth`` (plus
        ``filename_suffix`` for filepath-based types), preserving the
        dataset-centric directory layout so that the base catalog
        ``PartitionedDataset`` entries continue to discover all task outputs.
        Paths are written as POSIX strings (forward slashes) so that the
        generated ``catalog.yml`` files are valid on any operating system.

        Args:
            task_pth: Task leaf directory as a :class:`~pathlib.Path`, e.g.
                ``Path("sense_manage/task_0")``.

        Returns:
            Dict mapping dataset names to catalog entry dicts, ready to save as
            ``catalog.yml`` via the ``scenario_configs`` PartitionedDataset.
        """
        result = {}
        for name, entry in self.catalog.items():
            if not isinstance(entry, dict):
                continue
            override_meta = (entry.get("metadata") or {}).get("scenario_override")
            if override_meta is None:
                continue

            # Use PurePosixPath so catalog YAML values always use forward
            # slashes, even when this code runs on Windows.
            task_path = (PurePosixPath(entry.get("path", "")) / task_pth).as_posix()

            flatten = isinstance(override_meta, dict) and override_meta.get(
                "flatten", False
            )

            if flatten:
                inner = entry.get("dataset", {})
                inner_type = inner.get("type", entry["type"])
                new_entry = {k: v for k, v in inner.items() if k != "type"}
                new_entry["type"] = inner_type
                if inner_type in _FILEPATH_TYPES:
                    suffix = entry.get("filename_suffix", "")
                    new_entry["filepath"] = task_path + suffix
                else:
                    new_entry["path"] = task_path
            else:
                new_entry = {k: v for k, v in entry.items() if k != "metadata"}
                new_entry["path"] = task_path

            result[name] = new_entry
        return result

    def _build_single_partition(
        self: Self,
        pth: Path,
        params: dict,
        task_id: int,
    ) -> dict[str, dict]:
        """Build a Kedro configuration partition dict for a single task.

        Appends a ``task_<task_id>`` leaf to ``pth`` and returns a dict mapping
        config file paths to their contents.  Always emits a ``parameters``
        entry; also emits a ``catalog`` entry when
        :meth:`_build_single_catalog` returns a non-empty dict.

        Args:
            pth: Base directory for this task's output, down to (but not
                including) the ``task_N`` leaf.
            params: Parameter dict for this task.
            task_id: Zero-based task index (matches ``SLURM_ARRAY_TASK_ID``).

        Returns:
            Dict of ``{config_file_path: content}`` entries ready for merging
            into the full partitions dict returned by :meth:`build_configs`.
        """
        part_pth = pth / f"task_{task_id}"

        entries = {str(part_pth / "parameters"): params}

        catalog_data = self._build_single_catalog(part_pth)
        if catalog_data:
            entries[str(part_pth / "catalog")] = catalog_data

        return entries

    def build_configs(self: Self) -> dict[Path, dict]:
        """Generate the full set of Kedro configuration partition dicts.

        Calls :meth:`_build_param_dicts`, then wraps each ``(path, params)``
        pair with :meth:`_build_single_partition`.  Sets
        :attr:`n_tasks_generated` as a side-effect so the bash writer can
        generate the correct SLURM array range.

        Returns:
            Dict mapping each task's config file path to its content dict.
            Suitable for saving as a Kedro ``PartitionedDataset``.
        """
        paths, scen_params = self._build_param_dicts()
        self.n_tasks_generated = len(paths)

        parts = {}
        for task, (path, scen) in enumerate(zip(paths, scen_params)):
            cur_part = self._build_single_partition(pth=path, params=scen, task_id=task)
            parts.update(cur_part)
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
