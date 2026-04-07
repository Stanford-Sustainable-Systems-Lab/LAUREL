import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Self

import openturns as ot

from megaplug.scenarios.build import ScenarioBuilder
from megaplug.scenarios.read import ScenarioReader
from megaplug.utils.sensitivity import correl_dict_to_matrix

CATEGORICAL_DIST_NAME = "Categorical"


class SenseScenarioBuilder(ScenarioBuilder):
    """Build scenarios for the sensitivity truck model with discrete variables."""

    partition_level_names = (
        "run_name",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []

        vars = self.scen_params["variables"]
        params_categ = {
            k: v for k, v in vars.items() if v["dist"] == CATEGORICAL_DIST_NAME
        }
        params_sampl = {k: v for k, v in vars.items() if k not in params_categ}

        # Build categorical enumeration
        categ_names = list(params_categ.keys())
        categ_values = [v["params"] for v in params_categ.values()]
        categ_combos = [
            dict(zip(categ_names, combo)) for combo in product(*categ_values)
        ]

        # Build joint sampling distribution
        joint_dist = self.build_input_dist(
            vars=params_sampl,
            copulas=self.scen_params.get("copulas", None),
        )
        sequence = ot.SobolSequence(joint_dist.getDimension())
        sample_size = self.scen_params[
            "sample_size"
        ]  # Sobol' sequences work best on powers of 2
        experiment = ot.LowDiscrepancyExperiment(sequence, joint_dist, sample_size)
        samples = experiment.generate().asDataFrame()
        samples.columns = list(params_sampl.keys())

        # Get parameter dicts to override (must override from top-level key)
        if not all("path" in d for d in vars.values()):
            raise ValueError(
                f"{str(self.__class__)} must have paths for each variable."
            )
        top_keys = set([d["path"][0] for d in vars.values()])

        for pars_categ in categ_combos:
            for _, row in samples.iterrows():
                pars_sampl = row.to_dict()
                pars = pars_categ | pars_sampl

                scn = {k: deepcopy(self.params[k]) for k in top_keys}
                for k, v in pars.items():
                    self._set_param(scn, vars[k]["path"], v)

                paths.append(Path(self.display_name))
                scens.append(scn)
        return (paths, scens)

    @staticmethod
    def _set_param(d: dict, keys: list, v: object) -> None:
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = v

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
    """Read scenarios for the sensitivity truck model with discrete variables."""

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
