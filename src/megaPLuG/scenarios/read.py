import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Self

import pandas as pd

from .build import ScenarioBuilder, TestScenarioBuilder


class ScenarioReader(ABC):
    """This class sets the interface for scenario readers. These readers use the `kedro`
    partitioned datasets set up by the ScenarioBuilder classes and read them into
    combined datasets which are usable for post-hoc analysis."""

    builder: ScenarioBuilder
    metadata_level_names: tuple[str]
    scenario_name: str = "Scenario"

    def __init__(self: Self) -> None:
        meta_names = set(self.metadata_level_names)
        build_names = set(self.builder.partition_level_names)
        unmatched_names = meta_names - build_names
        if len(unmatched_names) > 0:
            raise RuntimeError(
                f"Some of the metadata names don't correspond with partition_level_names of the builder: {unmatched_names}"
            )

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

    @staticmethod
    def select_partitions(
        partitions: dict[Path, object], dirs: str | list[str]
    ) -> dict[Path, object]:
        """Select all partitions which are within any of the given directories."""
        if isinstance(dirs, str):
            dirs = [dirs]
        dirs = [Path(d) for d in dirs]
        candids = product(dirs, partitions)
        selected = {
            pth: partitions[pth] for dir, pth in candids if pth.is_relative_to(dir)
        }
        if len(selected) < 1:
            raise RuntimeError(
                "No partitions were identified in the given directories."
            )
        return selected

    def read_partitions(
        self: Self,
        partitions: dict[str, object],
        dirs: str | list[str],
    ) -> object:
        """Read data from the specific directories (dirs) within a PartitionDataset
        given by partitions.
        """
        parts = {Path(d): o for d, o in partitions.items()}
        parts = self.select_partitions(partitions=parts, dirs=dirs)
        names = {pth: self.name_scenario(pth) for pth in parts.keys()}
        metadata = {pth: self.extract_metadata(pth) for pth in parts.keys()}

        test_data = list(parts.values())[0]()  # Loading from the loader function
        if isinstance(test_data, pd.DataFrame):
            coll = self._collate_partitions_df(
                partitions=parts,
                names=names,
                metadata=metadata,
            )
        elif isinstance(test_data, dict):
            coll = {names[pth]: loader() for pth, loader in parts.items()}
        else:
            raise NotImplementedError(
                "ScenarioReader collation for this dataset type has not been implemented."
            )

        return coll

    def _collate_partitions_df(
        self: Self,
        partitions: dict[str, Callable],
        names: dict[str, str],
        metadata: dict[str, tuple],
    ) -> pd.DataFrame:
        """Collate partitions with dataframe data into a single long dataframe with
        keys given by the scenario name and selected metadata.
        """
        df_ls, key_ls = [], []
        for pth, load_func in partitions.items():
            df_ls.append(load_func())
            cur_key = (names[pth], *metadata[pth])
            key_ls.append(cur_key)

        names_ls = [self.scenario_name] + [*self.metadata_level_names]
        coll = pd.concat(df_ls, keys=key_ls, names=names_ls)
        colnames = coll.columns.tolist()
        coll = coll.reset_index(self.metadata_level_names)
        coll = coll.loc[:, colnames + [*self.metadata_level_names]]
        return coll

    def list_completed_partitions(
        self: Self,
        data_partitions: dict[str, object],
        dirs: str | list[str],
        config_partitions: dict[str, object] = None,
        incomplete: bool = False,
        report_type: str = "scenario",
    ) -> list[str | int]:
        """List the completed partitions by forcing the builder to build its configs,
        then counting within the given directory the number of completed tasks relative
        to the number of configs.

        Returns: A list of scenario names or task ids, depending on the report_type
        argument.
        """
        # Use the select partitions based on the builder's display name and get the set of paths
        data_partitions = {Path(d): o for d, o in data_partitions.items()}
        complete_parts = self.select_partitions(partitions=data_partitions, dirs=dirs)
        complete_parts = set(complete_parts.keys())
        report_set = complete_parts

        if incomplete:
            # Use the builder to generate all of its configs and take the set of paths
            config_partitions = {Path(d): o for d, o in config_partitions.items()}
            target_parts = self.select_partitions(
                partitions=config_partitions, dirs=dirs
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
