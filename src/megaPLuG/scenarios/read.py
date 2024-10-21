from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Self

from .build import ScenarioBuilder


class ScenarioReader(ABC):
    """This class sets the interface for scenario readers. These readers use the `kedro`
    partitioned datasets set up by the ScenarioBuilder classes and read them into
    combined datasets which are usable for post-hoc analysis."""

    builder: ScenarioBuilder

    def __init__(self: Self) -> None:
        pass

    @property
    @abstractmethod
    def builder(self: Self) -> ScenarioBuilder:
        """Get the ScenarioBuilder which this reader reads from."""
        pass  # Implement this in the concrete classes by setting the builder attribute

    @abstractmethod
    def get_level_data(self: Self, levels: tuple[str]) -> dict:
        """Get the data encoded in the level values."""
        pass  # TODO: This will receive a subset of the set_reporting_groups code

    @abstractmethod
    def get_level_names(self: Self, levels: tuple[str]) -> tuple[str]:
        """Format each of the partition level names."""
        pass  # TODO: This will receive a tuplified version of set_reporting_groups

    def concat_level_names(self: Self, *args: str, sep: str = ", ") -> str:
        """Concatenate level names into a single scenario name string."""
        concat = [str(level).replace("_", " ").title() for level in args]
        name = sep.join(concat)
        return name

    def name_scenarios(self: Self, partitions: dict[Path, object]) -> str:
        """Name the scenarios based on their partition paths."""
        for pth, obj in partitions.items():
            names = self.get_level_names(pth.parts)
            scen = self.concat_level_names(*names)
            partitions["scenario"] = scen
        return partitions

    def select_partitions(
        self: Self, partitions: dict[Path, object], dirs: str | list[str]
    ) -> dict[Path, object]:
        """Select all partitions which are within any of the given directories."""
        if isinstance(dirs, str):
            dirs = [dirs]
        candids = product(dirs, partitions)
        selected = {
            pth: partitions[pth] for dir, pth in candids if pth.is_relative_to(dir)
        }
        return selected

    def read_partitions(
        self: Self, partitions: dict[str, object], single: bool = False
    ):
        # TODO: We've got to get both the scenario name as the key and the formatted level names as well as the main dataset
        # Then, delegate reading of the specific type of data file to the relevant data type loader,
        # which will often be a pandas dataframe loader (see _read_partitions_df).
        pass

    def _read_partitions_df(self: Self, partitions: dict[str, object]):
        parts = self.read_partitions(partitions=partitions)
        # TODO: Add a pandas concat in here inspired by collate_scenario_partitions
        return parts

    def list_completed_partitions(report_type) -> list[str | int]:
        """List the completed partitions by forcing the builder to build its configs,
        then counting within the given directory the number of completed tasks relative
        to the number of configs.

        Returns: A list of scenario names or task ids, depending on the report_type
        argument.
        """
        # TODO: Use set operations to compute the incomplete scenarios
        pass
