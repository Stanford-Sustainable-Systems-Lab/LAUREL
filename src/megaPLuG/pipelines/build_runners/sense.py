import re
from copy import deepcopy
from pathlib import Path
from typing import Self

import openturns as ot

from megaPLuG.scenarios.build import ScenarioBuilder
from megaPLuG.scenarios.read import ScenarioReader


class SenseScenarioBuilder(ScenarioBuilder):
    """Build scenarios for the sensitivity truck model."""

    display_name = "sense"
    partition_level_names = (
        "run_name",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []

        joint_dist = self.build_input_dist(self.scen_params["variables"])
        sequence = ot.SobolSequence(joint_dist.getDimension())
        sample_size = self.scen_params[
            "sample_size"
        ]  # Sobol' sequences work best on powers of 2
        experiment = ot.LowDiscrepancyExperiment(sequence, joint_dist, sample_size)
        samples = experiment.generate().asDataFrame()
        samples.columns = list(self.scen_params["variables"].keys())

        for _, row in samples.iterrows():
            pars = row.to_dict()
            cur_vehs = deepcopy(self.params["vehicles"])
            cur_vehs["consump_rate_kwh_per_mi"]["values"][True][8] = pars[
                "consump_kwh_per_mile"
            ]
            cur_vehs["consump_rate_kwh_per_mi"]["values"][False][8] = pars[
                "consump_kwh_per_mile"
            ]
            cur_vehs["battery_capacity_kwh"]["values"][True][8] = pars["batt_cap_kwh"]
            cur_vehs["battery_capacity_kwh"]["values"][False][8] = pars["batt_cap_kwh"]
            cur_vehs["charge_soc_thresh"] = pars["charge_thresh_soc"]
            cur_vehs["dwell_min_soc_boost_frac"] = pars["min_soc_thresh"]
            cur_vehs["minimum_times"]["plug_in_mins"] = pars["time_to_initiate_mins"]
            cur_vehs["minimum_times"]["plug_out_mins"] = pars["time_to_initiate_mins"]

            cur_locs = deepcopy(self.params["locations"])
            cur_locs["max_power_kw"]["values"]["depot"] = pars["charge_speed_kw_depot"]
            cur_locs["max_power_kw"]["values"]["other"] = pars[
                "charge_speed_kw_enroute"
            ]

            cur_mngr = deepcopy(self.params["manage_charging"])
            num_mngr = pars["charge_management"]
            str_mngr = (
                "MinPowerChargingManager"
                if num_mngr == 1
                else "ImmediateChargingManager"
            )
            cur_mngr["charging_manager"] = str_mngr

            scn = {
                "vehicles": cur_vehs,
                "locations": cur_locs,
                "manage_charging": cur_mngr,
            }

            paths.append(Path(self.display_name))
            scens.append(scn)
        return (paths, scens)

    @staticmethod
    def build_input_dist(vars: dict) -> ot.JointDistribution:
        """Build the input distribution from the vars parameter dictionary."""
        marg_dists = []
        for var_name, dist_info in vars.items():
            dist_type = dist_info["dist"]
            dist_params = dist_info["params"]
            cur_dist = getattr(ot, dist_type)(*dist_params)
            cur_dist.setDescription([var_name])
            marg_dists.append(cur_dist)

        joint_dist = ot.JointDistribution(marg_dists)
        return joint_dist


class SenseScenarioReader(ScenarioReader):
    """Read scenarios for the sensitivity truck model."""

    builder = SenseScenarioBuilder
    metadata_level_names = ("task_id",)

    def extract_metadata(self: Self, path: Path) -> tuple:
        meta = self.get_metadata_values(path=path)
        task_id = int(re.search(r"(?<=task_)(\d+)", meta["task_id"]).group())
        return (task_id,)

    def name_scenario(self: Self, path: Path) -> str:
        meta = self.get_metadata_values(path=path)
        task_id = int(re.search(r"(?<=task_)(\d+)", meta["task_id"]).group())
        return task_id
