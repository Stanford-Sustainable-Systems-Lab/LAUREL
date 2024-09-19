import re
from itertools import product
from pathlib import Path

import pandas as pd


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
