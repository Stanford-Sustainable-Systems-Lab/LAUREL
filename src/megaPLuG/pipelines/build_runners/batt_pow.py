import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Self

from megaPLuG.scenarios.build import ScenarioBuilder
from megaPLuG.scenarios.read import ScenarioReader


class BatteryPowerScenarioBuilder(ScenarioBuilder):
    """Create scenarios which scan across battery sizes and charging powers."""

    partition_level_names = (
        "run_name",
        "batt_kwh",
        "depot_kw",
        "is_mandate_active",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []
        batt_levels = self.scen_params["battery_capacities_kwh"]
        dkw_levels = self.scen_params["charge_speed_kw_depot"]
        mand_levels = self.scen_params["is_mandate_active"]
        for batt, dkw, mand in product(batt_levels, dkw_levels, mand_levels):
            pth = Path(
                self.display_name, f"batt_{batt}", f"depot_{dkw}", f"mand_{mand}"
            )
            cur_vehs = deepcopy(self.params["vehicles"])
            cur_vehs["battery_capacity_kwh"] = batt

            cur_modes = deepcopy(self.params["charging_modes"])
            cur_modes["depot"]["max_power_kw"] = dkw

            cur_samp = deepcopy(self.params["build_sampling_totals"])
            cur_samp["filter_totals"]["filters"]["is_mandate_active"]["value_isin"] = [
                mand
            ]

            scn = {
                "vehicles": cur_vehs,
                "charging_modes": cur_modes,
                "build_sampling_totals": cur_samp,
            }
            paths.append(pth)
            scens.append(scn)
        return (paths, scens)


class BatteryPowerScenarioReader(ScenarioReader):
    """Read scenarios which scan across battery sizes and management strategies."""

    builder = BatteryPowerScenarioBuilder
    metadata_level_names = ("batt_kwh", "depot_kw", "is_mandate_active")

    def extract_metadata(self: Self, path: Path) -> tuple:
        meta = self.get_metadata_values(path=path)
        batt = int(re.search(r"(?<=batt_)(\d+)", meta["batt_kwh"]).group())
        dkw = int(re.search(r"(?<=depot_)(\d+)", meta["depot_kw"]).group())
        mand = bool(
            eval(re.search(r"(?<=mand_)(.+)", meta["is_mandate_active"]).group())
        )
        return (batt, dkw, mand)

    def name_scenario(self: Self, path: Path) -> str:
        meta = self.get_metadata_values(path=path)
        batt_kwh = re.search(r"(?<=batt_)(\d+)", meta["batt_kwh"]).group() + "kWh"
        dkw = re.search(r"(?<=depot_)(\d+)", meta["depot_kw"]).group() + "kW"
        mand = (
            "mandate_" + re.search(r"(?<=mand_)(.+)", meta["is_mandate_active"]).group()
        )
        return self.concat_name_components(batt_kwh, dkw, mand, sep=",\n")
