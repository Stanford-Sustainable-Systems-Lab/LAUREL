from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy as pt
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.special import expit


@dataclass
class ElectProbLocalizerConfig:
    """Configuration for ElectProbLocalizer.

    Columns:
      - loc_col: location class column (default "cluster_id").
      - veh_col: vehicle class column (default "primary_op_dist").
      - weight_col: P(L|V) column (default "p_lclass_g_vclass").
      - n_obs_col: total observations per (L,V) cell (default "n_obs").
      - n_electrified_col: electrified observations per (L,V) (default "n_electrified").
      - target_col: target P(E|V) (default "p_elect_g_vclass").

    Model/solver:
      - formula: Patsy formula. If None, uses main effects C(loc_col)+C(veh_col).
      - ridge_lambda: L2 penalty magnitude; intercept unpenalized when penalize_intercept=False.
      - penalize_intercept: whether to penalize intercept.
      - min_obs: minimum n_obs to include a row in GLM fitting.
      - bracket: initial bracket for per-vehicle calibration root finding.
      - max_bracket_expansions: how many times to expand bracket if root not bracketed.
    """

    # Column names
    loc_col: str = "cluster_id"
    veh_col: str = "primary_op_dist"
    weight_col: str = "p_lclass_g_vclass"
    n_obs_col: str = "n_obs"
    n_electrified_col: str = "n_electrified"
    target_col: str = "p_elect_g_vclass"

    # Model config
    formula: str | None = None
    ridge_lambda: float = 2.5
    penalize_intercept: bool = False
    min_obs: int = 1

    # Calibration config
    bracket: float = 25.0
    max_bracket_expansions: int = 4


class ElectProbLocalizer:
    """Fuse probability tables to obtain localized adoption probability P(E | L, V).

    Required columns in the input DataFrame (names configurable via ElectProbLocalizerConfig):
      - loc_col (e.g., "cluster_id")
      - veh_col (e.g., "primary_op_dist")
      - n_electrified_col
      - n_obs_col
      - weight_col (P(L|V))
      - target_col (P(E|V))

    Steps:
      1) Fit a (ridge-penalized) binomial GLM to sample proportions where n_obs >= min_obs.
      2) Predict linear predictors eta_hat for all (L,V) rows.
      3) For each vehicle class V, solve for a scalar delta_V such that
            sum_L P(L|V) * logistic(eta_hat_{L,V} + delta_V) = P(E|V)
         using a robust bracket expansion if necessary.
      4) Return a Series p_elect_g_vclass_loc indexed like the input DataFrame.

    Usage:
        cfg = ElectProbLocalizerConfig(loc_col="cluster_id", veh_col="primary_op_dist")
        fuser = ElectProbLocalizer(df, config=cfg)
        p_series = fuser.fit_transform()
        df["p_elect_g_vclass_loc"] = p_series
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: ElectProbLocalizerConfig | None = None,
        copy: bool = True,
    ) -> None:
        self.config = config or ElectProbLocalizerConfig()

        # Determine required columns based on config
        cfg = self.config
        required = [
            cfg.loc_col,
            cfg.veh_col,
            cfg.n_electrified_col,
            cfg.n_obs_col,
            cfg.weight_col,
            cfg.target_col,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing required columns: {missing}")

        # NaN validation for required columns
        nan_counts = df[required].isna().sum()
        bad_nan = nan_counts[nan_counts > 0]
        if not bad_nan.empty:
            details = {col: int(cnt) for col, cnt in bad_nan.items()}
            raise ValueError(
                "NaN values detected in required columns. Clean or impute before proceeding: "
                f"{details}"
            )

        # Work on a copy to avoid mutating caller unless copy=False
        self.df = df.copy(deep=True) if copy else df
        self._fitted = False
        self._results: sm.GLM | None = None
        self._params: pd.Series | None = None
        self._eta_hat: pd.Series | None = None
        self._delta: pd.Series | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self) -> ElectProbLocalizer:
        cfg = self.config
        df = self.df

        # Build working subset for fitting
        df_fit = df.loc[df[cfg.n_obs_col] >= cfg.min_obs].copy()
        if df_fit.empty:
            raise ValueError("No rows meet minimum observation threshold for fitting.")
        df_fit["y"] = df_fit[cfg.n_electrified_col].astype(float) / df_fit[
            cfg.n_obs_col
        ].astype(float)

        # Build formula dynamically if not provided
        formula = cfg.formula or f"y ~ C({cfg.loc_col}) + C({cfg.veh_col})"

        # Design matrices
        y_design, X = pt.dmatrices(formula, data=df_fit, return_type="dataframe")
        model = sm.GLM(
            y_design,
            X,
            family=sm.families.Binomial(),
            var_weights=df_fit[cfg.n_obs_col],
        )

        # Ridge penalty vector (statsmodels uses alpha for each param)
        alpha = np.full(shape=(X.shape[1],), fill_value=cfg.ridge_lambda, dtype=float)
        if not cfg.penalize_intercept and "Intercept" in X.columns:
            alpha[X.columns.get_loc("Intercept")] = 0.0

        results = model.fit_regularized(alpha=alpha, L1_wt=0.0)
        self._results = results
        self._params = results.params

        # Predict linear predictors for all rows (even those not used in fit)
        # Need to supply a dummy y to build design for the full df
        _, X_all = pt.dmatrices(formula, data=df.assign(y=0), return_type="dataframe")
        # Align columns (handle any dropped levels)
        common_cols = [c for c in results.params.index if c in X_all.columns]
        X_sel = X_all.loc[:, common_cols]
        eta_linear = results.predict(X_sel, which="linear")
        # Reindex to full DataFrame order
        eta_series = pd.Series(eta_linear, index=df.index, name="eta_hat")
        self._eta_hat = eta_series

        # Per-vehicle calibration
        delta = self._compute_deltas()
        self._delta = delta

        self._fitted = True
        return self

    def transform(self) -> pd.Series:
        if not self._fitted or self._eta_hat is None or self._delta is None:
            raise RuntimeError("Call fit() before transform().")
        df = self.df
        cfg = self.config
        p = expit(self._eta_hat + self._delta.loc[df[cfg.veh_col].values].values)
        return pd.Series(p, index=df.index, name="p_elect_g_vclass_loc")

    def fit_transform(self) -> pd.Series:
        return self.fit().transform()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_deltas(self) -> pd.Series:
        cfg = self.config
        df = self.df
        if self._eta_hat is None:
            raise RuntimeError("Eta (linear predictors) not computed.")

        # Group by vehicle class
        deltas = {}
        for vclass, grp in df.groupby(cfg.veh_col):
            eta_sub = self._eta_hat.loc[grp.index].values
            w_sub = grp[cfg.weight_col].values.astype(float)
            m_target = grp[cfg.target_col].iloc[0]

            # Explicit handling of NaNs or non-finite weights
            if np.isnan(w_sub).any():
                raise ValueError(
                    f"Encountered NaN weight(s) in {cfg.weight_col} for {cfg.veh_col}={vclass}. "
                    "These must be cleaned or imputed before calibration."
                )
            if (~np.isfinite(w_sub)).any():
                raise ValueError(
                    f"Encountered non-finite (inf/-inf) weight(s) for {cfg.veh_col}={vclass}."
                )
            if w_sub.sum() <= 0:
                raise ValueError(
                    f"Non-positive sum of weights for {cfg.veh_col}={vclass}. Cannot calibrate delta."
                )
            w_sub = w_sub / w_sub.sum()

            # Define root function
            def g(delta: float) -> float:
                return np.sum(w_sub * expit(eta_sub + delta)) - m_target

            # Initial bracket search
            a = -cfg.bracket
            b = cfg.bracket
            fa = g(a)
            fb = g(b)
            expansions = 0
            while fa * fb > 0 and expansions < cfg.max_bracket_expansions:
                # Expand symmetrically
                a *= 2
                b *= 2
                fa = g(a)
                fb = g(b)
                expansions += 1

            if fa * fb > 0:
                # Fallback: approximate shift via logit difference of means (approximation)
                # Compute current weighted mean probability
                p_curr = np.sum(w_sub * expit(eta_sub))
                # Stabilize extremes
                eps = 1e-8
                p_curr = np.clip(p_curr, eps, 1 - eps)
                m_t = np.clip(m_target, eps, 1 - eps)
                delta_est = np.log(m_t / (1 - m_t)) - np.log(p_curr / (1 - p_curr))
                deltas[vclass] = float(delta_est)
            else:
                try:
                    deltas[vclass] = float(brentq(g, a, b))
                except ValueError:
                    # Should not happen, but fallback to 0 shift
                    deltas[vclass] = 0.0
        return pd.Series(deltas, name="delta")

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def params_(self) -> pd.Series:
        if self._params is None:
            raise AttributeError("Model not yet fit: call fit().")
        return self._params

    @property
    def eta_hat_(self) -> pd.Series:
        if self._eta_hat is None:
            raise AttributeError("Model not yet fit: call fit().")
        return self._eta_hat

    @property
    def deltas_(self) -> pd.Series:
        if self._delta is None:
            raise AttributeError("Model not yet fit: call fit().")
        return self._delta


# Example usage:
# cfg = ElectProbLocalizerConfig(loc_col="cluster_id", veh_col="primary_op_dist")
# fuser = ElectProbLocalizer(cross_cls, config=cfg)
# cross_cls["p_elect_g_vclass_lclass_cls"] = fuser.fit_transform()
