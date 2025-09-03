from pathlib import Path

from .read import ScenarioReader


def write_scenario_partition(obj: object, params: dict) -> tuple[dict[str:object], str]:
    """Write out the partition for this scenario and dataset.

    Insert this into a kedro pipeline to enable saving out of data files to different
    partitions based on scenario.

    Args:
        params: the "results_partition" element in the configuration dictionary, passed
        straight from the kedro pipeline

    Returns: A kedro partition dictionary, with the directory as the key and the object
    to save as the value. The mode of saving should be managed from the catalog.yml
    """
    return {params["dir"]: obj}, "order_ensurer"


def read_scenario_partition(
    partitions: dict, params: dict, order_ensurer: str = None
) -> object:
    """Read in a single partition for this scenario and dataset.

    Insert this into a kedro pipeline to enable loading in of a file from a partition
    which you're also saving from.

    If you want to load in from multiple partitions at once and collate the results,
    then use `ScenarioReader.read_partitions()` instead.
    """
    dir = params["dir"]
    parts = {Path(d): o for d, o in partitions.items()}
    selected = ScenarioReader.select_partitions(partitions=parts, dirs=dir)
    if len(selected) > 1:
        raise RuntimeError("More than one partition identified in the given directory.")
    part_load_func = list(selected.values())[0]
    data = part_load_func()
    return data
