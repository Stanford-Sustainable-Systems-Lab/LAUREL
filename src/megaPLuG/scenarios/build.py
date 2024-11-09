from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np


class ScenarioBuilder(ABC):
    """This class sets the interface for scenario builders. These builders use
    the `kedro` configuration file system to create many runs with different parameters.
    The `SLURM_ARRAY_TASK_ID` environment variable within SLURM is used to sweep across
    them, or you can use a local `for` loop from the command line.
    """

    scen_param_key: str = "results_partition"
    n_tasks_generated: int = None

    def __init__(self, scen_params: dict, all_params: dict) -> None:
        self.scen_params = scen_params
        self.params = all_params

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Get the display name used to build folders and SLURM runs for this scenario."""
        pass  # Implement this in the concrete classes by setting the attribute display_name

    @property
    @abstractmethod
    def partition_level_names(self) -> tuple[str]:
        """Get the partition level names used to explain the partition levels."""
        pass  # Implement this in the concrete classes by setting the attribute partition_level_names

    @abstractmethod
    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        """Build the (filepath, parameter dict) pairs which define this scenario."""
        pass

    def _build_single_partition(
        self: Self,
        pth: Path,
        params: dict,
        task_id: int,
    ) -> dict[str:dict]:
        """Build a configuration partition for running a scenario.

        Args:
            pth: the path to be used for partitions of data files in this scenario
                set down to the single task level
            params: the parameters to be added to the scenario's config
            task_id: the number of the task within this scenario set

        Returns: dictionary giving the file path and configuration dict for a
        specific scenario.
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
        paths, scen_params = self._build_param_dicts()
        parts = {}
        for task, (path, scen) in enumerate(zip(paths, scen_params)):
            cur_part = self._build_single_partition(pth=path, params=scen, task_id=task)
            parts.update(cur_part)
        self.n_tasks_generated = len(parts)
        return parts

    @staticmethod
    def _get_random_seed(seed: int, max: int = 1000) -> int:
        """Get a random seed for a scenario run."""
        rng = np.random.default_rng(seed=seed)
        num = rng.integers(max)
        return int(num)


class TestScenarioBuilder(ScenarioBuilder):
    """Builds a test scenario set with a single partition."""

    display_name = "test"
    partition_level_names = ("run_name", "task_id")

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths = [Path(self.display_name)]
        scens = [{}]
        return (paths, scens)
