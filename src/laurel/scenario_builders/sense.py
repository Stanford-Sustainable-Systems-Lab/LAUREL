"""Scenario builder and reader for sensitivity analyses with mixed variable types.

Supports sensitivity designs that combine *categorical* variables (enumerated
discrete choices, e.g. charging-manager strategy) with *continuous* variables
sampled via a Sobol' quasi-random sequence (e.g. adoption rates, energy
consumption rates).  The full scenario space is the Cartesian product of all
categorical combinations and all Sobol samples, giving
``n_categ_combos × sample_size`` total scenarios.

Classes
-------
- :class:`SenseScenarioBuilder`: Constructs scenario parameter dicts for HPC
  batch execution.
- :class:`SenseScenarioReader`: Parses scenario paths back into metadata for
  cross-scenario aggregation.

Key design decisions
--------------------
- **Categorical × continuous product**: Crossing every categorical combination
  with every Sobol draw ensures that the full continuous parameter space is
  explored independently for each discrete configuration — important when
  categorical choices interact with continuous parameters (e.g. charging
  strategy interacts with charging speed).
- **Graceful empty-set handling**: If ``params_sampl`` is empty (all variables
  are categorical), ``samples_list = [{}]`` keeps the loop structure uniform
  and avoids passing a zero-dimension distribution to OpenTURNS.  If
  ``params_categ`` is empty (all variables are continuous), ``product(*[])``
  yields one empty tuple, so ``categ_combos = [{}]`` and the full Sobol sample
  is iterated without modification.
- **Path-based parameter override**: Each variable definition includes a
  ``path`` key — a list of dict keys that locates the variable inside the
  Kedro parameter hierarchy.  :meth:`SenseScenarioBuilder._set_param` navigates
  this path to mutate a deep-copied base parameter dict, ensuring no two
  scenarios share mutable state.
- **Copula structure**: Correlations among continuous variables (e.g. adoption
  rates that are jointly drawn from a Gaussian copula fit to NLR scenarios)
  are specified as a separate ``copulas`` block in the scenario config.
  Independent variables are assigned an ``IndependentCopula`` automatically,
  and all copulas are combined into a ``BlockIndependentCopula``.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.
"""

import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Self

import openturns as ot

from laurel.scenario_framework.build import ScenarioBuilder
from laurel.scenario_framework.read import ScenarioReader
from laurel.utils.sensitivity import correl_dict_to_matrix

CATEGORICAL_DIST_NAME = "Categorical"


class SenseScenarioBuilder(ScenarioBuilder):
    """Build scenario parameter dicts for sensitivity analyses with mixed variable types.

    Partitions the variables defined in ``scenario_params["variables"]`` into
    categorical variables (enumerated via :func:`itertools.product`) and
    continuous variables (sampled via a Sobol' low-discrepancy sequence).  The
    full scenario space is the Cartesian product of the two sets.
    """

    partition_level_names = (
        "run_name",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        """Build the list of scenario parameter dicts for this sensitivity design.

        Constructs one parameter dict per scenario by crossing all categorical
        combinations with all Sobol-sampled continuous draws:

        1. **Partition variables**: split ``scenario_params["variables"]`` into
           ``params_categ`` (``dist == "Categorical"``) and ``params_sampl``
           (all others).
        2. **Enumerate categoricals**: compute the Cartesian product of all
           categorical ``params`` lists via :func:`itertools.product`, yielding
           ``categ_combos`` — a list of ``{var_name: value}`` dicts.
        3. **Sample continuous variables**: if any continuous variables exist,
           build a :class:`~openturns.JointDistribution` (with optional copula
           structure) and draw ``sample_size`` points from a Sobol' sequence.
           If no continuous variables exist, use ``[{}]`` so the loop still
           executes once per categorical combination.
        4. **Assemble scenarios**: for each ``(categ_combo, sample_row)`` pair,
           deep-copy the relevant top-level Kedro parameter dicts and apply each
           variable value via :meth:`_set_param` following its ``path``.

        Returns:
            Two-tuple ``(paths, scens)`` where:

            - ``paths``: list of :class:`~pathlib.Path` objects, one per
              scenario, each set to ``Path(self.display_name)``.
            - ``scens``: list of parameter dicts, each containing deep-copied
              overrides for the top-level Kedro parameter keys touched by at
              least one variable in this design.

        Raises:
            ValueError: If any variable definition is missing a ``path`` key.
        """
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
        if params_sampl:
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
            samples_list = [row.to_dict() for _, row in samples.iterrows()]
        else:
            samples_list = [{}]

        # Get parameter dicts to override (must override from top-level key)
        if not all("path" in d for d in vars.values()):
            raise ValueError(
                f"{str(self.__class__)} must have paths for each variable."
            )
        top_keys = set([d["path"][0] for d in vars.values()])

        for pars_categ in categ_combos:
            for pars_sampl in samples_list:
                pars = pars_categ | pars_sampl

                scn = {k: deepcopy(self.params[k]) for k in top_keys}
                for k, v in pars.items():
                    self._set_param(scn, vars[k]["path"], v)

                paths.append(Path(self.display_name))
                scens.append(scn)
        return (paths, scens)

    @staticmethod
    def _set_param(d: dict, keys: list, v: object) -> None:
        """Assign a value at a nested location in a multi-level dictionary.

        Traverses ``d`` by successively indexing with each element of ``keys``
        except the last, then assigns ``v`` at the final key.  Mutates ``d``
        in place — callers should pass a deep copy if the original must be
        preserved.

        Args:
            d: The dictionary to modify.
            keys: Ordered list of keys defining the path from the top level to
                the target location.  Must have at least one element.
            v: Value to assign at ``d[keys[0]][keys[1]]...[keys[-1]]``.
        """
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = v

    @staticmethod
    def build_input_dist(
        vars: dict, copulas: list[dict] | None
    ) -> ot.JointDistribution:
        """Build the joint input distribution for continuous sensitivity variables.

        Constructs an :class:`~openturns.JointDistribution` from marginal
        distributions and an optional block-independent copula structure:

        1. For each variable in ``vars``, instantiate the named OpenTURNS
           distribution (e.g. ``ot.Beta``, ``ot.Uniform``) using the ``params``
           values in order of their definition, and label it with the variable
           name.
        2. If ``copulas`` is provided, build one copula per entry (currently
           ``NormalCopula`` and ``StudentCopula`` are supported).  Each copula
           entry specifies a ``correlation`` sub-dict parsed by
           :func:`~laurel.utils.sensitivity.correl_dict_to_matrix`.  Variables
           not assigned to any copula block are collected into an
           ``IndependentCopula``.  All copulas are combined into a
           ``BlockIndependentCopula``.
        3. If no copulas are provided, the marginals are assembled with no
           dependence structure (independent joint distribution).

        Args:
            vars: Ordered dict mapping variable names to distribution
                specifications.  Each value must have:

                - ``dist`` (``str``): OpenTURNS distribution class name.
                - ``params`` (``dict``): Keyword-argument values passed
                  positionally to the distribution constructor; order matters.
            copulas: List of copula specification dicts, each with a single
                key naming the copula type (``"NormalCopula"`` or
                ``"StudentCopula"``) and a value containing a ``correlation``
                sub-dict.  Pass ``None`` for independent marginals.

        Returns:
            :class:`~openturns.JointDistribution` combining the specified
            marginals and copula structure.

        Raises:
            ValueError: If a copula entry is not a dict, or if copula variable
                names are not a subset of the marginal variable names.
            NotImplementedError: If a copula type other than ``NormalCopula``
                or ``StudentCopula`` is requested.
        """
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
    """Read and index scenarios produced by :class:`SenseScenarioBuilder`.

    Parses the integer task ID from each scenario's path metadata, which is
    used downstream to order scenarios and join cross-scenario aggregations.
    """

    builder = SenseScenarioBuilder
    metadata_level_names = ("task_id",)

    def extract_metadata(self: Self, path: Path) -> tuple:
        """Extract the numeric task ID from a scenario path.

        Args:
            path: Scenario path as returned by the builder.

        Returns:
            One-tuple ``(task_id,)`` where ``task_id`` is the integer parsed
            from the ``task_<N>`` segment of the path metadata.
        """
        meta = self.get_metadata_values(path=path)
        task_id = int(re.search(r"(?<=task_)(\d+)", meta["task_id"]).group())
        return (task_id,)

    def name_scenario(self: Self, path: Path) -> str:
        """Return the integer task ID as the canonical scenario name.

        Args:
            path: Scenario path as returned by the builder.

        Returns:
            Integer task ID parsed from the path metadata.
        """
        meta = self.get_metadata_values(path=path)
        task_id = int(re.search(r"(?<=task_)(\d+)", meta["task_id"]).group())
        return task_id
