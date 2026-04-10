import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Self

from laurel.scenario_framework.build import ScenarioBuilder
from laurel.scenario_framework.read import ScenarioReader


class CalifClass8AdoptionScenarioBuilder(ScenarioBuilder):
    """Build scenarios for the California Class 8 truck model."""

    partition_level_names = (
        "run_name",
        "adopt_pct",
        "range_mi",
        "depot_kw",
        "enroute_kw",
        "charge_management",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []
        adopt_levels = self.scen_params["adopt_pcts"]
        range_levels = self.scen_params["range_miles"]
        depot_kw_levels = self.scen_params["depot_kw_levels"]
        enroute_kw_levels = self.scen_params["enroute_kw_levels"]
        charge_management_levels = self.scen_params["charging_managers"]

        iter = product(
            adopt_levels,
            range_levels,
            depot_kw_levels,
            enroute_kw_levels,
            charge_management_levels,
        )
        for adopt, range, dkw, erkw, mngr in iter:
            pth = Path(
                self.display_name,
                f"adopt_{int(adopt * 100)}",
                f"range_{range}",
                f"depot_{dkw}",
                f"enroute_{erkw}",
                mngr,
            )

            cur_tots = deepcopy(self.params["build_sampling_totals"])
            cur_tots["adoption_frac"] = adopt

            cur_vehs = deepcopy(self.params["vehicles"])
            consump_vals = cur_vehs["consump_rate_kwh_per_mi"]["values"]
            cur_vehs["battery_capacity_kwh"]["values"] = multiply_dict_leaves(
                consump_vals, range
            )

            cur_powers = deepcopy(self.params["locations"])
            cur_powers["max_power_kw"]["values"] = {
                "depot": dkw,
                "other": erkw,
            }
            cur_mngr = deepcopy(self.params["manage_charging"])
            cur_mngr["charging_manager"] = mngr

            scn = {
                "build_sampling_totals": cur_tots,
                "vehicles": cur_vehs,
                "locations": cur_powers,
                "manage_charging": cur_mngr,
            }

            paths.append(pth)
            scens.append(scn)
        return (paths, scens)


class CalifClass8AdoptionScenarioReader(ScenarioReader):
    """Read scenarios for the California Class 8 truck model."""

    builder = CalifClass8AdoptionScenarioBuilder
    metadata_level_names = (
        "adopt_pct",
        "range_mi",
        "depot_kw",
        "enroute_kw",
        "charge_management",
    )

    def extract_metadata(self: Self, path: Path) -> tuple:
        meta = self.get_metadata_values(path=path)
        adopt_pct = int(re.search(r"(?<=adopt_)(\d+)", meta["adopt_pct"]).group()) / 100
        range_mi = int(re.search(r"(?<=range_)(\d+)", meta["range_mi"]).group())
        depot_kw = int(re.search(r"(?<=depot_)(\d+)", meta["depot_kw"]).group())
        enroute_kw = int(re.search(r"(?<=enroute_)(\d+)", meta["enroute_kw"]).group())
        manage = re.search(
            r"(.+)(?=ChargingManager)", meta["charge_management"]
        ).group()
        return (adopt_pct, range_mi, depot_kw, enroute_kw, manage)

    def name_scenario(self: Self, path: Path) -> str:
        meta = self.get_metadata_values(path=path)
        adopt_pct = (
            str(int(re.search(r"(?<=adopt_)(\d+)", meta["adopt_pct"]).group()))
            + "% Adopt"
        )
        range_mi = re.search(r"(?<=range_)(\d+)", meta["range_mi"]).group() + "mi"
        depot_kw = re.search(r"(?<=depot_)(\d+)", meta["depot_kw"]).group() + "kw Home"
        enroute_kw = (
            re.search(r"(?<=enroute_)(\d+)", meta["enroute_kw"]).group() + "kw Away"
        )
        manage = re.search(
            r"(.+)(?=ChargingManager)", meta["charge_management"]
        ).group()
        return self.concat_name_components(
            adopt_pct, range_mi, depot_kw, enroute_kw, manage
        )


def multiply_dict_leaves(d, scalar):
    """Multiply all leaf values in a dictionary by a scalar."""
    if isinstance(d, dict):
        return {k: multiply_dict_leaves(v, scalar) for k, v in d.items()}
    else:
        # Leaf node - multiply by scalar
        return d * scalar
