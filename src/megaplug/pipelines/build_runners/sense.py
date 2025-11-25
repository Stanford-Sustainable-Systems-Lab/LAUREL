import re
from copy import deepcopy
from pathlib import Path
from typing import Self

import openturns as ot

from megaplug.scenarios.build import ScenarioBuilder
from megaplug.scenarios.read import ScenarioReader
from megaplug.utils.sensitivity import correl_dict_to_matrix


class SenseScenarioBuilder(ScenarioBuilder):
    """Build scenarios for the sensitivity truck model."""

    partition_level_names = (
        "run_name",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []

        joint_dist = self.build_input_dist(
            vars=self.scen_params["variables"],
            copulas=self.scen_params.get("copulas", None),
        )
        sequence = ot.SobolSequence(joint_dist.getDimension())
        sample_size = self.scen_params[
            "sample_size"
        ]  # Sobol' sequences work best on powers of 2
        experiment = ot.LowDiscrepancyExperiment(sequence, joint_dist, sample_size)
        samples = experiment.generate().asDataFrame()
        samples.columns = list(self.scen_params["variables"].keys())

        for _, row in samples.iterrows():
            pars = row.to_dict()

            cur_adopts = deepcopy(self.params["compute_adoption_totals"])
            cur_adopts["adoption_fracs"]["values"]["0-99 Miles"] = pars[
                "adopt_0_99_miles"
            ]
            cur_adopts["adoption_fracs"]["values"]["100-249 Miles"] = pars[
                "adopt_100_249_miles"
            ]
            cur_adopts["adoption_fracs"]["values"]["250-499 Miles"] = pars[
                "adopt_250_499_miles"
            ]
            cur_adopts["adoption_fracs"]["values"]["500+ Miles"] = pars[
                "adopt_500plus_miles"
            ]

            cur_vehs = deepcopy(self.params["vehicles"])
            cur_vehs["consump_rate_kwh_per_mi"] = pars["consump_rate_kwh_per_mi"]
            cur_vehs["soc_buffer_low"] = pars["soc_buffer_low"]

            cur_modes = deepcopy(self.params["charging_modes"])
            cur_modes["enroute"]["max_power_kw"] = pars["charge_speed_kw_enroute"]
            cur_modes["depot"]["max_power_kw"] = pars["charge_speed_kw_depot"]
            cur_modes["truck_stop"]["max_power_kw"] = pars["charge_speed_kw_truck_stop"]

            scn = {
                "compute_adoption_totals": cur_adopts,
                "vehicles": cur_vehs,
                "charging_modes": cur_modes,
            }

            paths.append(Path(self.display_name))
            scens.append(scn)
        return (paths, scens)

    @staticmethod
    def build_input_dist(
        vars: dict, copulas: list[dict] | None
    ) -> ot.JointDistribution:
        """Build the input distribution from the vars parameter dictionary."""
        marg_dists = []
        for var_name, dist_info in vars.items():
            dist_type = dist_info["dist"]
            dist_params = list(dist_info["params"].values())
            cur_dist = getattr(ot, dist_type)(*dist_params)
            cur_dist.setDescription([var_name])
            marg_dists.append(cur_dist)

        if copulas:
            var_set = set(vars.keys())
            cop_ls = []
            for cop_info in copulas:
                if not isinstance(cop_info, dict):
                    raise ValueError(
                        f"Copula list elements must be dictionaries. but got {type(cop_info)}"
                    )
                cop_type = list(cop_info.keys())[0]
                cop_args = cop_info[cop_type]
                if cop_type not in ["NormalCopula", "StudentCopula"]:
                    raise NotImplementedError(
                        f"Only correlation-based copulas are implemented, but {cop_type} was requested."
                    )
                cop_corr, var_names = correl_dict_to_matrix(cop_args["correlation"])
                if var_set.intersection(var_names) != set(var_names):
                    raise ValueError(
                        f"Variables included in copulas whose marginals are not defined: {var_names}"
                    )
                cur_cop = getattr(ot, cop_type)(cop_corr)
                cur_cop.setDescription(var_names)
                cop_ls.append(cur_cop)
                var_set = var_set.difference(var_names)

            if var_set:
                rem_cop = ot.IndependentCopula(len(var_set))
                rem_cop.setDescription(list(var_set))
                cop_ls.append(rem_cop)

            joint_cop = ot.BlockIndependentCopula(cop_ls)
        else:
            joint_cop = None

        joint_dist = ot.JointDistribution(marg_dists, joint_cop)
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
