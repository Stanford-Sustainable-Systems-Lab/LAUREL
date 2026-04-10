"""Abstract base classes for reading scenario-partitioned datasets into analysis DataFrames.

Provides :class:`ScenarioReader`, the abstract interface that concrete readers
must implement to collate Kedro ``PartitionedDataset`` partitions across
scenario runs (States of the World) into a single combined DataFrame.

Typical post-hoc analysis workflow
-----------------------------------
1. Instantiate a concrete reader with a list of scenario directories (or
   ``None`` to read all available partitions).
2. Call :meth:`ScenarioReader.read_partitions` with the Kedro partition dict
   to obtain a combined DataFrame labelled by scenario metadata.
3. Use :meth:`ScenarioReader.list_completed_partitions` to audit which tasks
   have finished before reading.

Key design decisions
--------------------
- **Lazy/eager dual path**: :meth:`ScenarioReader.read_partitions` supports
  ``lazy=True`` (Dask ``from_map``) for large post-hoc analyses that would
  exhaust memory if loaded eagerly, and ``lazy=False`` (pandas ``concat``) for
  small result sets or interactive exploration.
- **Multi-level column handling**: :meth:`ScenarioReader._collate_partition_df`
  pads metadata column names with empty strings when the DataFrame has a
  multi-level column index, so that metadata columns appear at the correct
  level without restructuring the DataFrame.
- **``metadata_level_names ⊆ partition_level_names`` validation**: the
  constructor checks this invariant at instantiation time, surfacing
  misconfiguration before any I/O is attempted.
- **Path-based metadata extraction**: scenario metadata (e.g. adoption-rate
  bin, energy-rate value) is encoded in the partition path by
  :class:`~laurel.scenario_framework.build.ScenarioBuilder`; concrete readers
  parse it via :meth:`extract_metadata` rather than storing it in the data
  files, keeping the files themselves schema-free.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import product, repeat
from pathlib import Path
from typing import Self

import dask.dataframe as dd
import pandas as pd

from .build import ScenarioBuilder, TestScenarioBuilder


class ScenarioReader(ABC):
    """Abstract base class for collating scenario-partitioned Kedro datasets.

    Concrete subclasses must implement three abstract members:

    - :attr:`builder` — the :class:`~laurel.scenario_framework.build.ScenarioBuilder`
      class whose ``partition_level_names`` defines the path structure.
    - :attr:`metadata_level_names` — the subset of ``partition_level_names``
      to extract as metadata columns in the combined DataFrame.
    - :meth:`extract_metadata` — parse a partition path to a metadata tuple.
    - :meth:`name_scenario` — convert a partition path to a human-readable label.

    Class attributes:
        builder: The :class:`~laurel.scenario_framework.build.ScenarioBuilder` class
            (not instance) associated with this reader.
        metadata_level_names: Tuple of path-level names to extract as DataFrame
            metadata columns; must be a subset of
            ``builder.partition_level_names``.
        scenario_name: Column name for the human-readable scenario label added
            by :meth:`_collate_partition_df`.  Defaults to ``"scenario"``.
    """

    builder: ScenarioBuilder
    metadata_level_names: tuple[str]
    scenario_name: str = "scenario"

    def __init__(self: Self, dirs: list[Path] | list[str] | None = None) -> None:
        """Initialise the reader and validate metadata/partition name consistency.

        Args:
            dirs: One or more directories to restrict reading to.  If a single
                string or ``Path`` is given, it is wrapped in a list.  Pass
                ``None`` to read all available partitions.

        Raises:
            RuntimeError: If any name in :attr:`metadata_level_names` is not
                present in ``builder.partition_level_names``.
        """
        meta_names = set(self.metadata_level_names)
        build_names = set(self.builder.partition_level_names)
        unmatched_names = meta_names - build_names
        if len(unmatched_names) > 0:
            raise RuntimeError(
                f"Some of the metadata names don't correspond with partition_level_names of the builder: {unmatched_names}"
            )

        if dirs is None:
            self.dirs = None
        else:
            if isinstance(dirs, str | Path):
                dirs_temp = [dirs]
            else:
                dirs_temp = dirs
            self.dirs: list[Path] | None = [Path(d) for d in dirs_temp]

    @property
    @abstractmethod
    def builder(self: Self) -> ScenarioBuilder:
        """The ScenarioBuilder class whose partition layout this reader expects."""
        pass  # Implement this in the concrete classes by setting the builder attribute

    @property
    @abstractmethod
    def metadata_level_names(self: Self) -> tuple[str]:
        """Ordered subset of ``builder.partition_level_names`` to use as metadata columns.

        Names must exactly match entries in
        :attr:`~laurel.scenario_framework.build.ScenarioBuilder.partition_level_names`
        of the associated builder.
        """
        pass

    @abstractmethod
    def extract_metadata(self: Self, path: Path) -> tuple:
        """Parse scenario-identifying metadata from a partition path.

        Return values must correspond in order with :attr:`metadata_level_names`.

        Args:
            path: ``Path`` object for a completed partition file.

        Returns:
            Tuple of metadata values (strings or scalars) in the same order as
            :attr:`metadata_level_names`.
        """
        pass

    @abstractmethod
    def name_scenario(self: Self, path: Path) -> str:
        """Convert a partition path to a human-readable scenario label.

        The helper :meth:`concat_name_components` is typically useful for
        formatting the label from path components.

        Args:
            path: ``Path`` object for a completed partition file.

        Returns:
            Human-readable scenario name string added to the combined DataFrame
            under the :attr:`scenario_name` column.
        """
        pass

    def concat_name_components(self: Self, *args: str, sep: str = ", ") -> str:
        """Join path-component strings into a formatted scenario name.

        Replaces underscores with spaces in each component, then joins with
        ``sep``.

        Args:
            *args: Path component strings (e.g. ``"high_adoption"``,
                ``"task_42"``).
            sep: Separator inserted between components.  Defaults to ``", "``.

        Returns:
            Formatted scenario name string (e.g. ``"high adoption, task 42"``).
        """
        concat = [str(level).replace("_", " ") for level in args]
        name = sep.join(concat)
        return name

    def get_metadata_values(self: Self, path: Path) -> dict[str, str]:
        """Extract the metadata-level path components from a partition path.

        Uses the positional indices of :attr:`metadata_level_names` within
        ``builder.partition_level_names`` to select the corresponding
        ``path.parts`` components.

        Args:
            path: ``Path`` object for a completed partition file.

        Returns:
            Dict mapping each metadata level name to its value from
            ``path.parts``.
        """
        idxs = {
            lev: self.builder.partition_level_names.index(lev)
            for lev in self.metadata_level_names
        }
        levels = {lev: path.parts[i] for lev, i in idxs.items()}
        return levels

    def select_partitions(
        self: Self,
        partitions: dict[Path, object],
    ) -> dict[Path, object]:
        """Filter a Kedro partition dict to entries under ``self.dirs``.

        Delegates to :meth:`select_partitions_static` with ``self.dirs``.

        Args:
            partitions: Full Kedro ``PartitionedDataset`` dict mapping path
                strings to loader callables.

        Returns:
            Filtered dict containing only the partitions whose paths are
            relative to one of the directories in ``self.dirs``.

        Raises:
            RuntimeError: If no matching partitions are found.
        """
        return ScenarioReader.select_partitions_static(
            partitions=partitions,
            dirs=self.dirs,
        )

    @staticmethod
    def select_partitions_static(
        partitions: dict[Path | str, object],
        dirs: list[Path] | None = None,
    ) -> dict[Path, object]:
        """Filter a partition dict to entries whose paths are under any of ``dirs``.

        If ``dirs`` is ``None``, all partitions are returned unchanged.  Uses
        ``Path.is_relative_to`` to test containment so both absolute and
        relative paths are handled consistently.

        Args:
            partitions: Kedro ``PartitionedDataset`` dict (path → loader).
            dirs: List of directories to restrict to, or ``None`` to return
                all partitions.

        Returns:
            Filtered dict of matching ``{Path: loader}`` entries.

        Raises:
            RuntimeError: If no partitions match any of the given directories.
        """
        part_dict = {Path(d): o for d, o in partitions.items()}
        if dirs is None:
            selected = part_dict
        else:
            candids = product(dirs, part_dict)
            selected = {
                pth: part_dict[pth] for dir, pth in candids if pth.is_relative_to(dir)
            }

        if len(selected) < 1:
            raise RuntimeError(
                "No partitions were identified in the given directories."
            )
        return selected

    def read_partitions(
        self: Self,
        partitions: dict[str, object],
        lazy: bool = False,
    ) -> object:
        """Load and collate scenario partitions into a single combined dataset.

        Filters the partition dict to the directories in ``self.dirs``, then
        for each matching partition calls the Kedro loader function, attaches
        metadata and scenario-name columns, and concatenates all results.

        For ``pd.DataFrame`` partitions, supports both eager (``pandas.concat``)
        and lazy (``dask.dataframe.from_map``) execution.  For ``dict``
        partitions, only eager loading is supported.

        Args:
            partitions: Full Kedro ``PartitionedDataset`` dict mapping path
                strings to zero-argument loader callables.
            lazy: If ``True``, return a Dask DataFrame constructed via
                ``from_map`` so that loading is deferred until ``.compute()``
                is called.  Only valid for ``pd.DataFrame`` partitions.
                Defaults to ``False``.

        Returns:
            Combined dataset.  For ``pd.DataFrame`` partitions: a
            ``pd.DataFrame`` (``lazy=False``) or ``dd.DataFrame``
            (``lazy=True``) with additional columns for each
            :attr:`metadata_level_names` entry and for :attr:`scenario_name`.
            For ``dict`` partitions: a dict keyed by scenario name.

        Raises:
            RuntimeError: If no partitions are found within ``self.dirs``.
            NotImplementedError: If ``lazy=True`` is requested for dict
                partitions, or if the partition data type is not supported.
        """
        part_dict = self.select_partitions(partitions=partitions)
        tups = [
            (self.name_scenario(pth), self.extract_metadata(pth), part)
            for pth, part in part_dict.items()
        ]
        names, metas, parts = zip(*tups)

        test_data = parts[0]()  # Loading from the loader function
        if isinstance(test_data, pd.DataFrame):
            if lazy:
                coll = dd.from_map(
                    self._collate_partition_df,
                    parts,
                    names,
                    metas,
                    metadata_level_names=self.metadata_level_names,
                    scenario_name=self.scenario_name,
                )
            else:
                coll_map = map(
                    self._collate_partition_df,
                    parts,
                    names,
                    metas,
                    repeat(self.metadata_level_names),
                    repeat(self.scenario_name),
                )
                coll = pd.concat(coll_map, axis=0)
        elif isinstance(test_data, dict):
            if lazy:
                raise NotImplementedError("Lazy dictionaries not yet implemented.")
            else:
                coll = {nm: pt() for nm, mt, pt in tups}
        else:
            raise NotImplementedError(
                "ScenarioReader collation for this dataset type has not been implemented."
            )

        return coll

    @staticmethod
    def _collate_partition_df(
        partition: Callable,
        name: str,
        metadata: tuple,
        metadata_level_names: tuple[str, ...],
        scenario_name: str,
        columns: list[object] | None = None,
    ) -> pd.DataFrame:
        """Load one partition and attach scenario metadata and name columns.

        Designed to be called via ``map`` or ``dask.dataframe.from_map``
        across all selected partitions.  Handles multi-level column indices by
        padding metadata column name tuples with empty strings so they align
        with the existing column level structure.

        Args:
            partition: Zero-argument callable that returns the partition's
                ``pd.DataFrame`` (the Kedro loader function).
            name: Human-readable scenario label (output of
                :meth:`~ScenarioReader.name_scenario`).
            metadata: Tuple of metadata values corresponding to
                ``metadata_level_names`` (output of
                :meth:`~ScenarioReader.extract_metadata`).
            metadata_level_names: Ordered tuple of metadata column names.
            scenario_name: Name of the scenario label column to add.
            columns: If provided, only these columns are retained in the
                output.  Pass ``None`` to keep all columns including all
                metadata and scenario columns.

        Returns:
            ``pd.DataFrame`` with metadata columns and the scenario column
            appended (or filtered to ``columns`` if specified).
        """
        # Load the dataframe using the Kedro-provided partition loader
        df = partition()

        # Add metadata columns, if requested
        ## Augment metadata_level_names with empty levels for multi-indexed columns
        if df.columns.nlevels > 1:
            add_ls = [""] * (df.columns.nlevels - 1)
            meta_names_ext = [tuple([meta] + add_ls) for meta in metadata_level_names]
        else:
            meta_names_ext = [*metadata_level_names]

        ## Get requested metadata columns
        if columns is not None:
            meta_cols = set(columns).intersection(meta_names_ext)
        else:
            meta_cols = set(meta_names_ext)

        meta_idx = [
            idx for idx, value in enumerate(meta_names_ext) if value in meta_cols
        ]
        meta_cols_add = [meta_names_ext[i] for i in meta_idx]
        meta_vals_add = [metadata[i] for i in meta_idx]
        meta_add = zip(meta_cols_add, meta_vals_add)

        ## Add requested metadata columns
        for col, val in meta_add:
            df[col] = val

        # Add scenario column, if requested
        if columns is None or scenario_name in columns:
            df[scenario_name] = name

        # Select chosen columns
        if columns is not None:
            df = df[columns]

        return df

    def list_completed_partitions(
        self: Self,
        data_partitions: dict[str, object],
        config_partitions: dict[str, object] | None = None,
        incomplete: bool = False,
        report_type: str = "scenario",
    ) -> list[str] | list[int]:
        """List completed (or incomplete) scenario partitions within ``self.dirs``.

        Compares the set of data partitions that already exist on disk against
        the full set of config partitions to determine which tasks have
        finished.  When ``incomplete=True``, returns the complement — the
        tasks that are still pending.

        Args:
            data_partitions: Kedro ``PartitionedDataset`` dict for the output
                dataset (e.g. ``dwells_with_charging_partition``).  Only
                partitions within ``self.dirs`` are considered.
            config_partitions: Kedro ``PartitionedDataset`` dict for the
                generated configs (used only when ``incomplete=True`` to
                determine the full target set).  May be ``None`` when
                ``incomplete=False``.
            incomplete: If ``True``, return tasks that are in ``config_partitions``
                but not yet in ``data_partitions``.  Defaults to ``False``
                (return completed tasks).
            report_type: Format of the return values.  ``"scenario"`` returns
                human-readable scenario names (via :meth:`name_scenario`);
                ``"task"`` returns integer task IDs parsed from the
                ``task_N`` path component.

        Returns:
            Sorted list of scenario name strings (``report_type="scenario"``)
            or integer task IDs (``report_type="task"``).

        Raises:
            RuntimeError: If no partitions are found within ``self.dirs``.
            NotImplementedError: If ``report_type`` is not ``"scenario"`` or
                ``"task"``.
        """
        # Use the select partitions based on the builder's display name and get the set of paths
        complete_parts = self.select_partitions(partitions=data_partitions)
        complete_parts = set(complete_parts.keys())
        report_set = complete_parts

        if incomplete:
            # Use the builder to generate all of its configs and take the set of paths
            target_parts = self.select_partitions(partitions=config_partitions)
            target_parts = [pth.parent for pth in target_parts.keys()]
            target_parts = set(target_parts)

            report_set = target_parts - complete_parts

        if report_type == "scenario":
            report_ls = [self.name_scenario(pth) for pth in report_set]
        elif report_type == "task":
            strs = [re.search(r"(?<=task_)\d+", str(s)).group() for s in report_set]
            report_ls = [int(s) for s in strs]
        else:
            raise NotImplementedError()
        report_ls.sort()
        return report_ls


class TestScenarioReader(ScenarioReader):
    """Minimal concrete reader for the single-partition test scenario.

    Pairs with :class:`~laurel.scenario_framework.build.TestScenarioBuilder`.
    Extracts no metadata and labels every partition ``"Test"``.
    """

    builder = TestScenarioBuilder
    metadata_level_names = ()

    def extract_metadata(self: Self, path: Path) -> tuple:
        """Return an empty tuple (no metadata levels defined for test scenarios).

        Args:
            path: Partition path (unused).

        Returns:
            Empty tuple ``()``.
        """
        return ()

    def name_scenario(self: Self, path: Path) -> str:
        """Return the fixed label ``"Test"`` for every partition.

        Args:
            path: Partition path (unused).

        Returns:
            The string ``"Test"``.
        """
        return "Test"
