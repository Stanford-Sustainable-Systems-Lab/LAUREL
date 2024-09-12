import functools
import re
from collections.abc import Callable
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def generate_configs(func: Callable[[dict], tuple[list[Path], list[dict]]]):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        paths, scen_params = func(*args, **kwargs)
        parts = {}
        for task, (path, scen) in enumerate(zip(paths, scen_params)):
            cur_part = build_config_partition(
                pth=path,
                params=scen,
                slurm_task_id=task,
            )
            parts.update(cur_part)
        return parts

    return wrapper_decorator


def build_config_partition(
    pth: Path,
    params: dict,
    slurm_task_id: int,
) -> dict[str:dict]:
    """Build a configuration partition for running a scenario.

    Args:
        pth: the path to be used for partitions of data files in this scenario
            set down to the single task level
        params: the parameters to be added to the scenario's config
        slurm_task_id: the number of the slurm task within this scenario set

    Returns: dictionary giving the file path and configuration dict for a
    specific scenario.
    """
    part_pth = pth / f"task_{slurm_task_id}"
    # Add the partition path to this scenario's config
    result_partition = {
        "dir": str(part_pth),
        "level_names": params["results_partition"]["level_names"],
    }
    params.update({"results_partition": result_partition})

    # Add this scenario's config to the partitions
    param_file = str(part_pth / "parameters")
    return {param_file: params}


def get_random_seed(seed: int, max: int = 1000) -> int:
    """Get a random seed for a scenario run."""
    rng = np.random.default_rng(seed=seed)
    num = rng.integers(max)
    return int(num)


def write_scenario_partition(obj: object, params: dict) -> dict[str:object]:
    """Write out the partition for this scenario and dataset.

    Insert this into a kedro pipeline to enable saving out of data files to different
    partitions based on scenario.

    Args:
        params: the "results_partition" element in the configuration dictionary, passed
        straight from the kedro pipeline

    Returns: A kedro partition dictionary, with the directory as the key and the object
    to save as the value. The mode of saving should be managed from the catalog.yml
    """
    return {params["dir"]: obj}


def read_scenario_partition(partitions: dict, params: dict) -> pd.DataFrame:
    """Read in a single partition for this scenario and dataset.

    Insert this into a kedro pipeline to enable loading in of a file from a partition
    which you're also saving from.

    If you want to load in from multiple partitions at once and collate the results,
    then use `collate_scenario_partitions` instead.
    """
    dir = params["dir"]
    partitions = {Path(k): v for k, v in partitions.items()}
    selected = _select_partitions(partitions, dir)
    if len(selected) > 1:
        raise RuntimeError(
            "More than one partition identified in the given directories."
        )
    if len(selected) < 1:
        raise RuntimeError("No partitions were identified in the given directories.")
    part_load_func = list(selected.values())[0]
    df = part_load_func()
    return df


def collate_scenario_partitions(partitions: dict, params: dict) -> pd.DataFrame:
    """Read in dataframes from select partitions and concatenate."""
    dirs = params["dir"]
    partitions = {Path(k): v for k, v in partitions.items()}
    selected = _select_partitions(partitions, dirs)

    if len(selected) < 1:
        raise RuntimeError("No partitions were identified in the given directories.")

    key_ls = [k.parts for k in selected.keys()]  # Yielding tuples of directory levels
    df_ls = [load_func() for load_func in selected.values()]
    coll = pd.concat(df_ls, keys=key_ls, names=params["level_names"])
    return coll


def _select_partitions(
    partitions: dict[Path, object], dirs: str | list[str]
) -> dict[Path, object]:
    """Select all partitions which are within any of the given directories."""
    if isinstance(dirs, str):
        dirs = [dirs]
    candids = product(dirs, partitions)
    selected = {pth: partitions[pth] for dir, pth in candids if pth.is_relative_to(dir)}
    return selected


def list_completed_tasks(pth: str) -> list[int]:
    """List the completed tasks within a kedro partitioned data directory."""
    pth = Path(pth)
    task_paths = list(pth.rglob("*task_*"))
    task_files = [p.name for p in task_paths]
    task_ids = [int(re.search(r"task_(\d+)", f)[1]) for f in task_files]
    task_ids.sort()
    return task_ids


@generate_configs
def build_test_configs(params: dict) -> tuple[list[Path], list[dict]]:
    """Build test scenario."""
    scen_params = params["scenario_params"]
    scen_name = scen_params["name"]
    paths = [Path(scen_name)]
    scens = [{}]
    return (paths, scens)


def generate_scenario_configs(params: dict) -> dict:
    """Call the appropriate scenario configuration builder.

    This function is meant to be called from a kedro pipeline directly.

    Args:
        params: the whole "parameters" input dictionary, usually passed directly from
            a kedro pipeline input.

    Returns: The dictionary of scenario configuration partitions.
    """
    scen_name = params["scenario_params"]["name"]
    if scen_name == "test":
        func = build_test_configs
    else:
        raise NotImplementedError(
            f"Scenario config generator for {scen_name} scenarios not yet implemented."
        )
    parts = func(params)
    return parts
