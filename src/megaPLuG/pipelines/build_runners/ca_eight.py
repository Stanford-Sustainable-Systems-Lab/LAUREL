import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Self

from megaPLuG.scenarios.build import ScenarioBuilder
from megaPLuG.scenarios.read import ScenarioReader


class CalifClass8ScenarioBuilder(ScenarioBuilder):
    """Build scenarios for the California Class 8 truck model."""

    partition_level_names = (
        "run_name",
        "range_mi",
        "depot_kw",
        "enroute_kw",
        "charge_management",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []
        range_levels = self.scen_params["range_miles"]
        depot_kw_levels = self.scen_params["depot_kw_levels"]
        enroute_kw_levels = self.scen_params["enroute_kw_levels"]
        charge_management_levels = self.scen_params["charging_managers"]

        iter = product(
            range_levels, depot_kw_levels, enroute_kw_levels, charge_management_levels
        )
        for range, dkw, erkw, mngr in iter:
            pth = Path(
                self.display_name,
                f"range_{range}",
                f"depot_{dkw}",
                f"enroute_{erkw}",
                mngr,
            )

            cur_vehs = deepcopy(self.params["vehicles"])
            consump_vals = cur_vehs["consump_rate_kwh_per_mi"]["values"]
            cur_vehs["battery_capacity_kwh"]["values"] = {
                True: {8: range * consump_vals[True][8]},
                False: {8: range * consump_vals[False][8]},
            }

            cur_powers = deepcopy(self.params["locations"])
            cur_powers["max_power_kw"]["values"] = {
                "depot": dkw,
                "other": erkw,
            }
            cur_mngr = deepcopy(self.params["manage_charging"])
            cur_mngr["charging_manager"] = mngr

            scn = {
                "vehicles": cur_vehs,
                "locations": cur_powers,
                "manage_charging": cur_mngr,
            }

            paths.append(pth)
            scens.append(scn)
        return (paths, scens)


class CalifClass8ScenarioReader(ScenarioReader):
    """Read scenarios for the California Class 8 truck model."""

    builder = CalifClass8ScenarioBuilder
    metadata_level_names = ("range_mi", "depot_kw", "enroute_kw", "charge_management")

    def extract_metadata(self: Self, path: Path) -> tuple:
        meta = self.get_metadata_values(path=path)
        range_mi = int(re.search(r"(?<=range_)(\d+)", meta["range_mi"]).group())
        depot_kw = int(re.search(r"(?<=depot_)(\d+)", meta["depot_kw"]).group())
        enroute_kw = int(re.search(r"(?<=enroute_)(\d+)", meta["enroute_kw"]).group())
        manage = re.search(
            r"(.+)(?=ChargingManager)", meta["charge_management"]
        ).group()
        return (range_mi, depot_kw, enroute_kw, manage)

    def name_scenario(self: Self, path: Path) -> str:
        meta = self.get_metadata_values(path=path)
        range_mi = re.search(r"(?<=range_)(\d+)", meta["range_mi"]).group() + "mi"
        depot_kw = re.search(r"(?<=depot_)(\d+)", meta["depot_kw"]).group() + "kw Home"
        enroute_kw = (
            re.search(r"(?<=enroute_)(\d+)", meta["enroute_kw"]).group() + "kw Away"
        )
        manage = re.search(
            r"(.+)(?=ChargingManager)", meta["charge_management"]
        ).group()
        return self.concat_name_components(range_mi, depot_kw, enroute_kw, manage)
