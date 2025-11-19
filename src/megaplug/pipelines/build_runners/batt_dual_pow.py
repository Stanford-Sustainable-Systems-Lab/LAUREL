import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Self

from megaplug.scenarios.build import ScenarioBuilder
from megaplug.scenarios.read import ScenarioReader


class BatteryDualPowerScenarioBuilder(ScenarioBuilder):
    """Create scenarios which scan across battery sizes and charging powers."""

    partition_level_names = (
        "run_name",
        "batt_kwh",
        "depot_kw",
        "tstop_kw",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []
        batt_levels = self.scen_params["battery_capacities_kwh"]
        dkw_levels = self.scen_params["charge_speed_kw_depot"]
        tkw_levels = self.scen_params["charge_speed_kw_truck_stop"]
        ekw_level = self.scen_params["charge_speed_kw_enroute"]
        for batt, dkw, tkw in product(batt_levels, dkw_levels, tkw_levels):
            pth = Path(
                self.display_name, f"batt_{batt}", f"depot_{dkw}", f"tstop_{tkw}"
            )
            cur_vehs = deepcopy(self.params["vehicles"])
            op_segs = list(cur_vehs["battery_capacity_kwh"]["values"].keys())
            for seg in op_segs:
                cur_vehs["battery_capacity_kwh"]["values"][seg][8] = batt

            cur_modes = deepcopy(self.params["charging_modes"])
            cur_modes["depot"]["max_power_kw"] = dkw
            cur_modes["truck_stop"]["max_power_kw"] = tkw
            cur_modes["enroute"]["max_power_kw"] = ekw_level

            scn = {
                "vehicles": cur_vehs,
                "charging_modes": cur_modes,
            }
            paths.append(pth)
            scens.append(scn)
        return (paths, scens)


class BatteryDualPowerScenarioReader(ScenarioReader):
    """Read scenarios which scan across battery sizes and management strategies."""

    builder = BatteryDualPowerScenarioBuilder
    metadata_level_names = ("batt_kwh", "depot_kw", "tstop_kw")

    def extract_metadata(self: Self, path: Path) -> tuple:
        meta = self.get_metadata_values(path=path)
        batt = int(re.search(r"(?<=batt_)(\d+)", meta["batt_kwh"]).group())
        dkw = int(re.search(r"(?<=depot_)(\d+)", meta["depot_kw"]).group())
        tkw = int(re.search(r"(?<=tstop_)(\d+)", meta["tstop_kw"]).group())
        return (batt, dkw, tkw)

    def name_scenario(self: Self, path: Path) -> str:
        meta = self.get_metadata_values(path=path)
        batt_kwh = re.search(r"(?<=batt_)(\d+)", meta["batt_kwh"]).group() + "kWh"
        dkw = re.search(r"(?<=depot_)(\d+)", meta["depot_kw"]).group() + "kW depot"
        tkw = re.search(r"(?<=tstop_)(\d+)", meta["tstop_kw"]).group() + "kW tstop"
        return self.concat_name_components(batt_kwh, dkw, tkw, sep=",\n")
