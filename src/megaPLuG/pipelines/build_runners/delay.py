import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Self

from megaPLuG.scenarios.build import ScenarioBuilder
from megaPLuG.scenarios.read import ScenarioReader


class DelayScenarioBuilder(ScenarioBuilder):
    """Build scenarios for the California Class 8 truck model."""

    display_name = "delay"
    partition_level_names = (
        "run_name",
        "batt_cap",
        "depot_kw",
        "enroute_kw",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []
        batt_cap_levels = self.scen_params["batt_cap_kwh"]
        depot_kw_levels = self.scen_params["depot_kw_levels"]
        enroute_kw_levels = self.scen_params["enroute_kw_levels"]

        iter = product(batt_cap_levels, depot_kw_levels, enroute_kw_levels)
        for batt, dkw, erkw in iter:
            pth = Path(
                self.display_name,
                f"batt_cap_{batt}",
                f"depot_{dkw}",
                f"enroute_{erkw}",
            )

            cur_vehs = deepcopy(self.params["vehicles"])
            cur_vehs["battery_capacity_kwh"]["values"] = {
                True: {8: batt},
                False: {8: batt},
            }

            cur_powers = deepcopy(self.params["charging_modes"])
            cur_powers["depot"]["max_power_kw"] = dkw
            cur_powers["enroute"]["max_power_kw"] = erkw

            scn = {
                "vehicles": cur_vehs,
                "charging_modes": cur_powers,
            }

            paths.append(pth)
            scens.append(scn)
        return (paths, scens)


class DelayScenarioReader(ScenarioReader):
    """Read scenarios for the California Class 8 truck model."""

    builder = DelayScenarioBuilder
    metadata_level_names = ("batt_cap", "depot_kw", "enroute_kw")

    def extract_metadata(self: Self, path: Path) -> tuple:
        meta = self.get_metadata_values(path=path)
        batt_cap = int(re.search(r"(?<=batt_cap_)(\d+)", meta["batt_cap"]).group())
        depot_kw = int(re.search(r"(?<=depot_)(\d+)", meta["depot_kw"]).group())
        enroute_kw = int(re.search(r"(?<=enroute_)(\d+)", meta["enroute_kw"]).group())
        return (batt_cap, depot_kw, enroute_kw)

    def name_scenario(self: Self, path: Path) -> str:
        meta = self.get_metadata_values(path=path)
        batt_cap = re.search(r"(?<=batt_cap_)(\d+)", meta["batt_cap"]).group() + "kWh"
        depot_kw = re.search(r"(?<=depot_)(\d+)", meta["depot_kw"]).group() + "kw Home"
        enroute_kw = (
            re.search(r"(?<=enroute_)(\d+)", meta["enroute_kw"]).group() + "kw Away"
        )
        return self.concat_name_components(batt_cap, depot_kw, enroute_kw)
