import functools
import re
from collections.abc import Callable
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
    result_partition = {"dir": str(part_pth)}
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


def collate_scenario_partitions(parts: dict, params: dict) -> pd.DataFrame:
    """Read in dataframes from select partitions and concatenate."""
    df_ls = []
    key_ls = []
    for part_key, part_load_func in parts.items():
        key_split = Path(part_key).parts
        if key_split[0] in params["include_partitions"]:
            cur_key = list(key_split)
            key_ls.append(tuple(cur_key))
            df_cur = part_load_func().reset_index()
            df_ls.append(df_cur)
    coll = pd.concat(df_ls, keys=key_ls, names=params["level_names"])
    coll.index = coll.index.droplevel(-1)
    return coll


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
