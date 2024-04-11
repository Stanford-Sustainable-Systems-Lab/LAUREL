"""Charging PGM
by Fletcher Passow
February 2024
"""

import arviz as az
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.ops.indexing import Vindex

from .pgm import PGM


class DistanceTimePGM(PGM):
    def __init__(self, params: dict):
        super().__init__(params)
        self.params.update({"alpha_dim": 5})

    @staticmethod
    def model(params: dict, n_obs: int):
        voc_probs = numpyro.sample(
            "voc_probs",
            dist.Dirichlet(jnp.ones(params["n_vocs"]) * params["vocation_conc_prior"]),
        )
        # Here, we learn something about the relative scales of the different values across vocations
        alpha_prior_dist = dist.Exponential(params["gamma_rate"]).expand(
            (params["alpha_dim"],)
        )
        alpha_concs = numpyro.sample("alpha_concs", alpha_prior_dist)
        alpha_rates = numpyro.sample("alpha_rates", alpha_prior_dist)
        with numpyro.plate("voc_plate", size=params["n_vocs"]):
            mix_dist = dist.Categorical(
                probs=jnp.ones(params["n_mix_components"]) / params["n_mix_components"]
            )

            def tile_mix(a):
                tile_ls = [params["n_mix_components"]] + [1] * a.ndim
                return jnp.tile(jnp.expand_dims(a, 0), tile_ls)

            comp_dist = dist.Gamma(
                concentration=tile_mix(alpha_concs),
                rate=tile_mix(alpha_rates),
            ).to_event(1)
            alpha_dist = dist.MixtureSameFamily(
                mixing_distribution=mix_dist, component_distribution=comp_dist
            )
            # TODO: Consider adding alpha to the veh_day_plate
            alpha = numpyro.sample("alpha", alpha_dist)
            alpha = jnp.clip(alpha, params["min_alpha"])

            concs = numpyro.sample("concs", dist.Exponential(params["gamma_rate"]))
            rates = numpyro.sample("rates", dist.Exponential(params["gamma_rate"]))

        with numpyro.plate("veh_day_plate", size=n_obs):
            # Local variables.
            voc = numpyro.sample(
                "voc", dist.Categorical(voc_probs), infer={"enumerate": "parallel"}
            )
            time_dist = dist.Dirichlet(concentration=Vindex(alpha)[..., voc, 0:3])
            times = numpyro.sample("times", time_dist)
            shift_dist = dist.Beta(
                concentration1=Vindex(alpha)[..., voc, 3],
                concentration0=Vindex(alpha)[..., voc, 4],
            )
            day_shift = numpyro.sample("day_shift", shift_dist)

            far_dist = dist.Gamma(concentration=concs[voc], rate=rates[voc])
            numpyro.sample("far", far_dist)

            # Deterministic schedule calculation
            if params["output_schedule"]:
                min_break_offset = (
                    params["min_active_session_length_hrs"] / params["total_hrs"]
                )
                break_shift = numpyro.sample(
                    "break_shift",
                    dist.Uniform(low=min_break_offset, high=1 - min_break_offset),
                )
                time_diffs = jnp.array(
                    [
                        Vindex(times)[..., 0],
                        break_shift * Vindex(times)[..., 1],
                        Vindex(times)[..., 2],
                        (1 - break_shift) * Vindex(times)[..., 1],
                    ]
                ).T
                time_cum = jnp.cumsum(time_diffs, axis=-1)
                day_shift = jnp.expand_dims(day_shift, -1)
                sched = jnp.mod(time_cum + day_shift, 1) * params["total_hrs"]
                numpyro.deterministic("sched", sched)
            # TODO: If transitioning back to NumPyro modeling, then implement a variable which tracks whether the schedule is electrifiable. We can then condition on it to return only electrifiable samples.

    def get_preds_df(self, preds: dict) -> pd.DataFrame:
        """Convert output predictions into a DataFrame."""
        idata = az.from_numpyro(
            posterior_predictive=preds,
            dims=self.params["pred_dims"],
            coords=self.params["pred_coords"],
        )
        pp = idata["posterior_predictive"]
        for var, arr in pp.data_vars.items():
            n_dims_to_add = self.params["max_var_dims"] - len(arr.dims)
            for _ in range(n_dims_to_add):
                pp[var] = pp[var].expand_dims(dim={"null_type": 1}, axis=-1)
        df = idata.to_dataframe(include_index=False)
        df.index = df.index.set_names("scen_id")
        df = df.drop(columns=["chain", "draw"])
        df.columns = pd.MultiIndex.from_tuples(
            df.columns, names=["var", "veh_day_id", "type"]
        )
        df = df.stack(["veh_day_id"])
        df.columns = [f"{i[0]}_{i[1]}" if i[1] != 0 else i[0] for i in df.columns]
        return df
