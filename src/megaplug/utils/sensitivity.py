from typing import Self

import openturns as ot
import pandas as pd
import seaborn as sns
from openturns import viewer

from megaplug.scenarios.build import ScenarioBuilder


class SensitivityAnalysis:
    """Perform a sensitivity analysis using a OpenTURNS."""

    def __init__(
        self: Self,
        scens: pd.DataFrame,
        output_col: str,
        input_vars: dict[str, dict],
        builder: ScenarioBuilder,
    ) -> None:
        """Create the OpenTURNS samples and joint distribution."""
        inputs = scens.loc[:, list(input_vars.keys())]
        self.inputs = ot.Sample.BuildFromDataFrame(inputs)
        outputs = scens.loc[:, [output_col]]
        self.outputs = ot.Sample.BuildFromDataFrame(outputs)
        self.joint_dist = builder.build_input_dist(input_vars)

    def fit_metamodel(self: Self, verbose: bool = True) -> None:
        """Fit the metamodel and, if verbose is True, print the diagnostics."""
        pce_estim = ot.FunctionalChaosAlgorithm(
            self.inputs, self.outputs, self.joint_dist
        )
        pce_estim.run()
        self.metamodel = pce_estim.getResult()

        if verbose:
            print(f"Metamodel Residuals: {self.metamodel.getResiduals()}")
            print(f"Metamodel Rel. Errors: {self.metamodel.getRelativeErrors()}")

            mod = self.metamodel.getMetaModel()
            val = ot.MetaModelValidation(self.outputs, mod(self.inputs))
            graph = val.drawValidation()
            viewer.View(graph)

    def calculate_sobols(self: Self) -> pd.DataFrame:
        """Calculate the first- and total-order Sobol' indices."""
        sobols = ot.FunctionalChaosSobolIndices(self.metamodel)
        input_dim = self.joint_dist.getDimension()
        first_order = [sobols.getSobolIndex(i) for i in range(input_dim)]
        total_order = [sobols.getSobolTotalIndex(i) for i in range(input_dim)]
        self.sobols = pd.DataFrame(
            {
                "Input Factor": list(self.joint_dist.getDescription()),
                "first": first_order,
                "total": total_order,
            }
        )
        return self.sobols

    def plot_sobols(self: Self) -> sns.FacetGrid:
        """Plot the Sobol' indices."""
        plot_sobols = self.sobols.melt(
            id_vars=["Input Factor"], var_name="order", value_name="Sensitivity Index"
        )
        plot_sobols["order"] = pd.Categorical(
            plot_sobols["order"], categories=["total", "first"], ordered=True
        )
        g = sns.catplot(
            data=plot_sobols,
            y="Input Factor",
            x="Sensitivity Index",
            hue="order",
            kind="bar",
            dodge=False,
        )
        for ax in g.axes.flat:
            ax.set_xlim(0, 1.0)
        return g


def correl_dict_to_matrix(correl_dict):
    """Convert a lower-triangular correlation dictionary into an OpenTURNS correlation matrix."""
    var_names = list(correl_dict.keys())
    dim = len(var_names)
    corr_mat = ot.CorrelationMatrix(dim)
    for i in range(dim):
        corr_mat[i, i] = 1.0
    for i, row_name in enumerate(var_names):
        row_entries = correl_dict.get(row_name, {})
        for j, col_name in enumerate(var_names[:i]):
            value = row_entries.get(col_name)
            if value is None:
                value = correl_dict.get(col_name, {}).get(row_name)
            if value is None:
                raise KeyError(
                    f"Missing correlation coefficient for pair ({row_name}, {col_name})."
                )
            corr_mat[i, j] = float(value)
            corr_mat[j, i] = float(value)
    return corr_mat, var_names


def correl_matrix_to_dict(corr_mat, var_names=None):
    """Convert an OpenTURNS correlation matrix into a lower-triangular correlation dictionary."""
    if var_names is None:
        var_names = [f"var_{i}" for i in range(corr_mat.getDimension())]
    corr_dict = {}
    for i, row_name in enumerate(var_names):
        row_entries = {}
        for j in range(i):
            row_entries[var_names[j]] = float(corr_mat[i, j])
        corr_dict[row_name] = row_entries
    return corr_dict
