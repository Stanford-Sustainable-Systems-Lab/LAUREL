import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Self

from megaPLuG.scenarios.build import ScenarioBuilder
from megaPLuG.scenarios.read import ScenarioReader


class BatteryManageScenarioBuilder(ScenarioBuilder):
    """Create scenarios which scan across battery sizes and management strategies."""

    display_name = "batt_man"
    partition_level_names = ("run_name", "batt_kwh", "charge_management", "task_id")

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []
        batt_levels = self.scen_params["battery_capacities_kwh"]
        manage_levels = self.scen_params["charging_managers"]
        for batt, mngr in product(batt_levels, manage_levels):
            pth = Path(self.display_name, f"batt_{batt}", mngr)
            cur_vehs = deepcopy(self.params["vehicles"])
            cur_vehs["battery_capacity_kwh"]["values"][True][8] = batt
            cur_mngr = deepcopy(self.params["manage_charging"])
            cur_mngr["charging_manager"] = mngr
            scn = {
                "vehicles": cur_vehs,
                "manage_charging": cur_mngr,
            }
            paths.append(pth)
            scens.append(scn)
        return (paths, scens)


class BatteryManageScenarioReader(ScenarioReader):
    """Read scenarios which scan across battery sizes and management strategies."""

    builder = BatteryManageScenarioBuilder
    metadata_level_names = ("batt_kwh", "charge_management")

    def extract_metadata(self: Self, path: Path) -> tuple:
        meta = self.get_metadata_values(path=path)
        batt = int(re.search(r"(?<=batt_)(\d+)", meta["batt_kwh"]).group())
        manage = re.search(
            r"(.+)(?=ChargingManager)", meta["charge_management"]
        ).group()
        return (batt, manage)

    def name_scenario(self: Self, path: Path) -> str:
        meta = self.get_metadata_values(path=path)
        batt_kwh = re.search(r"(?<=batt_)(\d+)", meta["batt_kwh"]).group() + "kWh"
        manage = re.search(
            r"(.+)(?=ChargingManager)", meta["charge_management"]
        ).group()
        return self.concat_name_components(batt_kwh, manage)
