"""Sensitivity analysis helpers using polynomial chaos expansion via OpenTURNS.

Provides :class:`SensitivityAnalysis` for computing first- and total-order
Sobol' indices from the N-SoW experimental design, and two helper functions
for converting between the lower-triangular correlation dictionary format used
in YAML configuration and the ``openturns.CorrelationMatrix`` format.

The Sobol' indices quantify how much of the variance in a scalar output (e.g.
peak substation load) is attributable to each input parameter, allowing
researchers to rank the importance of adoption-rate, charging-power, and
battery-reserve assumptions.

Key design decisions
--------------------
- **Polynomial chaos metamodel**: Sobol' indices are estimated from a
  functional-chaos polynomial expansion of the model response rather than
  by direct Monte Carlo, because the N-sample budget is too small for
  reliable direct estimation.  The metamodel approach also provides a
  validation metric (relative residual error) to flag poor fits.
- **Correlation dictionary format**: The YAML-friendly lower-triangular dict
  (only off-diagonal entries listed, indexed by variable name) is converted
  to/from the ``openturns.CorrelationMatrix`` object so that the copula
  correlation structure can be defined in config files without numeric
  indexing.
"""

from typing import Self

import openturns as ot
import pandas as pd
import seaborn as sns
from openturns import viewer

from laurel.scenarios.build import ScenarioBuilder


class SensitivityAnalysis:
    """Polynomial-chaos-based Sobol' sensitivity analysis using OpenTURNS.

    Wraps the OpenTURNS ``FunctionalChaosAlgorithm`` workflow:

    1. Convert scenario inputs and outputs to ``ot.Sample`` objects.
    2. Fit a polynomial chaos expansion metamodel via :meth:`fit_metamodel`.
    3. Extract first- and total-order Sobol' indices via :meth:`calculate_sobols`.
    4. Visualise results via :meth:`plot_sobols`.

    Args (constructor):
        scens: DataFrame of scenario results, one row per State of the World.
        output_col: Name of the scalar output column to analyse (e.g.
            ``"peak_load_95th_kw"``).
        input_vars: Dict mapping input variable names to their distribution
            specs (forwarded to ``ScenarioBuilder.build_input_dist``).
        builder: ``ScenarioBuilder`` instance used to construct the joint input
            distribution.
    """

    def __init__(
        self: Self,
        scens: pd.DataFrame,
        output_col: str,
        input_vars: dict[str, dict],
        builder: ScenarioBuilder,
    ) -> None:
        """Initialise OpenTURNS sample objects and build the joint input distribution."""
        inputs = scens.loc[:, list(input_vars.keys())]
        self.inputs = ot.Sample.BuildFromDataFrame(inputs)
        outputs = scens.loc[:, [output_col]]
        self.outputs = ot.Sample.BuildFromDataFrame(outputs)
        self.joint_dist = builder.build_input_dist(input_vars)

    def fit_metamodel(self: Self, verbose: bool = True) -> None:
        """Fit a functional-chaos polynomial expansion to the scenario input/output data.

        Runs ``ot.FunctionalChaosAlgorithm`` and stores the result as
        ``self.metamodel``.  If ``verbose=True``, prints the residual and
        relative error and displays a validation scatter plot.

        Args:
            verbose: If ``True``, print metamodel fit diagnostics and show a
                validation scatter plot.  Defaults to ``True``.
        """
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
        """Compute first- and total-order Sobol' indices from the fitted metamodel.

        Returns:
            DataFrame with columns ``["Input Factor", "first", "total"]``,
            one row per input variable.

        Raises:
            AttributeError: If :meth:`fit_metamodel` has not been called first.
        """
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
        """Render a horizontal bar chart of first- and total-order Sobol' indices.

        Returns:
            Seaborn ``FacetGrid`` with x-axis fixed to [0, 1].

        Raises:
            AttributeError: If :meth:`calculate_sobols` has not been called first.
        """
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
    """Convert a lower-triangular correlation dict to an ``openturns.CorrelationMatrix``.

    The dict format is::

        {
            "var_b": {"var_a": 0.6},
            "var_c": {"var_a": 0.3, "var_b": 0.5},
        }

    Only the lower triangle (and optionally either direction) needs to be provided;
    the function mirrors values to fill the upper triangle.

    Args:
        correl_dict: Nested dict mapping row variable → column variable →
            correlation coefficient.  Variable order is inferred from the top-level
            key order.

    Returns:
        Tuple of ``(ot.CorrelationMatrix, var_names)`` where ``var_names`` is the
        list of variable names in matrix index order.

    Raises:
        KeyError: If any off-diagonal pair is missing from the dict in both
            directions.
    """
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
    """Convert an ``openturns.CorrelationMatrix`` to a lower-triangular correlation dict.

    Args:
        corr_mat: An ``openturns.CorrelationMatrix`` object.
        var_names: Optional list of variable name strings, in the same order as
            the matrix rows/columns.  If ``None``, names default to
            ``"var_0"``, ``"var_1"``, etc.

    Returns:
        Lower-triangular dict in the format accepted by :func:`correl_dict_to_matrix`.
    """
    if var_names is None:
        var_names = [f"var_{i}" for i in range(corr_mat.getDimension())]
    corr_dict = {}
    for i, row_name in enumerate(var_names):
        row_entries = {}
        for j in range(i):
            row_entries[var_names[j]] = float(corr_mat[i, j])
        corr_dict[row_name] = row_entries
    return corr_dict
