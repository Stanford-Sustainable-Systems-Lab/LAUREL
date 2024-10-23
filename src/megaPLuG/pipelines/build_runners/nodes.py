"""
This is a boilerplate pipeline 'build_runners'
generated using Kedro 0.18.13
"""

import inspect
import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Self

from megaPLuG.scenarios.build import (  # noqa: F401
    ScenarioBuilder,
    TestScenarioBuilder,
)
from megaPLuG.scenarios.read import ScenarioReader


class BatteryManageScenarioBuilder(ScenarioBuilder):
    """Create scenarios which scan across battery sizes and management strategies."""

    display_name = "batt_man"
    partition_level_names = ("run_name", "batt_set", "manage_set", "task_id")

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []
        batt_levels = self.scen_params["battery_capacities_kwh"]
        manage_levels = self.scen_params["charging_managers"]
        for batt, mngr in product(batt_levels, manage_levels):
            pth = Path(self.display_name, f"batt_{batt}", mngr)
            cur_vehs = deepcopy(self.params["vehicles"])
            cur_vehs["battery_capacity_kwh"]["values"][True][8] = batt
            cur_mngr = deepcopy(self.params["profiles_from_dwells"])
            cur_mngr["charging_manager"] = mngr
            scn = {
                "vehicles": cur_vehs,
                "profiles_from_dwells": cur_mngr,
            }
            paths.append(pth)
            scens.append(scn)
        return (paths, scens)


class BatteryManageScenarioReader(ScenarioReader):
    """Read scenarios which scan across battery sizes and management strategies."""

    builder = BatteryManageScenarioBuilder
    metadata_names = ("batt_kwh", "charge_management")

    def extract_metadata(self: Self, path: Path) -> tuple:
        batt_str, manage_str = path.parts[1], path.parts[2]
        batt = int(re.search(r"(?<=batt_)(\d+)", batt_str).group())
        manage = re.search(r"(.+)(?=ChargingManager)", manage_str).group()
        return (batt, manage)

    def name_scenario(self: Self, path: Path) -> str:
        batt_str, manage_str = path.parts[1], path.parts[2]
        batt_kwh = re.search(r"(?<=batt_)(\d+)", batt_str).group() + "kWh"
        manage = re.search(r"(.+)(?=ChargingManager)", manage_str).group()
        return self.concat_name_components(batt_kwh, manage)


def generate_scenario_configs(scen_params: dict, all_params: dict) -> dict:
    """Call the appropriate scenario configuration builder.

    This function is meant to be called from a kedro pipeline directly.

    Args:
        scen_params: just the item in the "parameters" dictionary dedicated to giving
            parameters for scenario building (e.g. builder name)
        all_params: the whole "parameters" input dictionary, usually passed directly from
            a kedro pipeline input.

    Returns: The dictionary of scenario configuration partitions. Usually this would be
    saved to a `kedro` partitioned dataset.
    """
    bldr_map = {}
    base_cls = ScenarioBuilder
    for name, obj in globals().items():
        if inspect.isclass(obj) and issubclass(obj, base_cls) and obj is not base_cls:
            bldr_map.update({name: obj})

    bldr_name = scen_params["builder"]
    try:
        builder_cls = bldr_map[bldr_name]
    except KeyError:
        raise NotImplementedError(f"Scenario builder {bldr_name} not imported.")
    builder = builder_cls(scen_params=scen_params, all_params=all_params)
    parts = builder.build_configs()
    return (parts, builder)
