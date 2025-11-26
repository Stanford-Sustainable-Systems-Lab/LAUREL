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
    """This class sets the interface for scenario readers. These readers use the `kedro`
    partitioned datasets set up by the ScenarioBuilder classes and read them into
    combined datasets which are usable for post-hoc analysis."""

    builder: ScenarioBuilder
    metadata_level_names: tuple[str]
    scenario_name: str = "scenario"

    def __init__(self: Self, dirs: list[Path] | list[str] | None = None) -> None:
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
        """Get the ScenarioBuilder which this reader reads from."""
        pass  # Implement this in the concrete classes by setting the builder attribute

    @property
    @abstractmethod
    def metadata_level_names(self: Self) -> tuple[str]:
        """Get the metadata element names. These must form a subset of the self.builder
        partition_level_names attribute.
        """
        pass

    @abstractmethod
    def extract_metadata(self: Self, path: Path) -> tuple:
        """Get the metadata encoded in the partition path.

        These must correspond in ordering with `self.metadata_level_names`.
        """
        pass  # TODO: This will receive a subset of the set_reporting_groups code

    @abstractmethod
    def name_scenario(self: Self, path: Path) -> str:
        """Name the scenario based on its partition path.

        It is likely that `self.concat_name_components` will be useful for final
        formatting.
        """
        pass

    def concat_name_components(self: Self, *args: str, sep: str = ", ") -> str:
        """Concatenate level names into a single scenario name string."""
        concat = [str(level).replace("_", " ") for level in args]
        name = sep.join(concat)
        return name

    def get_metadata_values(self: Self, path: Path) -> dict[str, str]:
        """Get the parts from the current path which correspond to the metadata_levels."""
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
        """Select all partitions which are within any of the given directories."""
        return ScenarioReader.select_partitions_static(
            partitions=partitions,
            dirs=self.dirs,
        )

    @staticmethod
    def select_partitions_static(
        partitions: dict[Path | str, object],
        dirs: list[Path] | None = None,
    ) -> dict[Path, object]:
        """Select all partitions which are within any of the given directories."""
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
        """Read data from the specific directories (dirs) within a PartitionDataset
        given by partitions.
        """
        part_dict = {Path(d): o for d, o in partitions.items()}
        part_dict = self.select_partitions(partitions=part_dict, dirs=self.dirs)
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
        """Collate a dataframe partition with scenario columns."""
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
        """List the completed partitions by forcing the builder to build its configs,
        then counting within the given directory the number of completed tasks relative
        to the number of configs.

        Returns: A list of scenario names or task ids, depending on the report_type
        argument.
        """
        # Use the select partitions based on the builder's display name and get the set of paths
        data_parts = {Path(d): o for d, o in data_partitions.items()}
        complete_parts = self.select_partitions(partitions=data_parts, dirs=self.dirs)
        complete_parts = set(complete_parts.keys())
        report_set = complete_parts

        if incomplete:
            # Use the builder to generate all of its configs and take the set of paths
            config_parts = {Path(d): o for d, o in config_partitions.items()}
            target_parts = self.select_partitions(
                partitions=config_parts, dirs=self.dirs
            )
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
    """Read test scenarios."""

    builder = TestScenarioBuilder
    metadata_level_names = ()

    def extract_metadata(self: Self, path: Path) -> tuple:
        return ()

    def name_scenario(self: Self, path: Path) -> str:
        return "Test"
