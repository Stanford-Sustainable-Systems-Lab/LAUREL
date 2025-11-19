import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Self

from megaplug.scenarios.build import ScenarioBuilder
from megaplug.scenarios.read import ScenarioReader


class RangeScenarioBuilder(ScenarioBuilder):
    """Create scenarios which scan across battery sizes and charging powers."""

    partition_level_names = (
        "run_name",
        "range_mi",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []
        range_levels = self.scen_params["vehicle_range_mi"]
        for (range,) in product(range_levels):
            pth = Path(self.display_name, f"range_{range}")
            cur_vehs = deepcopy(self.params["vehicles"])
            op_segs = list(cur_vehs["battery_capacity_kwh"]["values"].keys())
            ecr = self.scen_params["consump_rate_kwh_per_mi"]
            for seg in op_segs:
                cur_vehs["consump_rate_kwh_per_mi"]["values"][seg][8] = ecr
                cur_vehs["battery_capacity_kwh"]["values"][seg][8] = range * ecr

            cur_modes = deepcopy(self.params["charging_modes"])
            for mode, pow in self.scen_params["charge_speed_kw"].items():
                cur_modes[mode]["max_power_kw"] = pow

            cur_evals = deepcopy(self.params["summarize_vehicles"])
            for name, thresh in self.scen_params["electrifiability_thresholds"].items():
                cur_evals["thresholds"][name] = thresh

            scn = {
                "vehicles": cur_vehs,
                "charging_modes": cur_modes,
                "summarize_vehicles": cur_evals,
            }
            paths.append(pth)
            scens.append(scn)
        return (paths, scens)


class RangeScenarioReader(ScenarioReader):
    """Read scenarios which scan across battery sizes and management strategies."""

    builder = RangeScenarioBuilder
    metadata_level_names = ("range_mi",)

    def extract_metadata(self: Self, path: Path) -> tuple:
        meta = self.get_metadata_values(path=path)
        batt = int(re.search(r"(?<=range_)(\d+)", meta["range_mi"]).group())
        return (batt,)

    def name_scenario(self: Self, path: Path) -> str:
        meta = self.get_metadata_values(path=path)
        range_mi = re.search(r"(?<=range_)(\d+)", meta["range_mi"]).group() + "mi"
        return self.concat_name_components(range_mi, sep=",\n")
