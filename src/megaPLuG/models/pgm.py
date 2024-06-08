"""Full PGM model for Fleet Charging
by Fletcher Passow
August 2023
"""

import abc
from collections.abc import Callable
from functools import partial
from warnings import warn

import graphviz
import numpyro
import pandas as pd
from jax import random
from numpyro.handlers import condition, seed, trace
from numpyro.infer import MCMC, NUTS, Predictive, init_to_feasible, init_to_value


class PGM(abc.ABC):
    """Base class for PGMs built off of numpyro, with fit, predict, and sample management."""

    mcmc = None
    samples = None
    params = None

    @abc.abstractmethod
    def __init__(self, params: dict):
        self.params = params
        self.set_samples({})

    def get_model(self, n_obs: int = 1) -> Callable:
        return partial(self.model, params=self.params, n_obs=n_obs)

    @staticmethod
    @abc.abstractmethod
    def model():
        pass

    def fit(
        self,
        data: dict = None,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        verbose: bool = True,
        rand: int = 0,
        init_values: dict = None,
        *args,
        **kwargs,
    ):
        """Fit the model.

        args and kwargs passed through to get_model

        While numpyro does not require this, we assume that all fit data has
        the same dimension, creating an observed values plate by default.

        """
        first_data_key = list(data.keys())[0]
        n_obs_fit = data[first_data_key].shape[0]
        model = self.get_model(n_obs=n_obs_fit, *args, **kwargs)
        if data is not None:
            model = condition(model, data=data)
        if verbose:
            self.print_shapes()

        if init_values is not None:
            init_strat = init_to_value(values=init_values)
        else:
            init_strat = init_to_feasible()

        self.mcmc = MCMC(
            NUTS(model, init_strategy=init_strat),
            num_warmup=num_warmup,
            num_samples=num_samples,
        )
        with numpyro.validation_enabled():
            self.mcmc.run(random.PRNGKey(rand))
        if verbose:
            self.mcmc.print_summary()
        # Note: Setting group_by_chain to False ensures that sample dimensions
        # will work with predict.
        self.samples = self.mcmc.get_samples(group_by_chain=False)
        self.avail_fitted_samples = num_samples
        return self

    def predict(
        self,
        return_sites: list[str],
        data_single: dict = None,
        data_sample: dict = None,
        num_samples: int = None,
        num_obs: int = 1,
        rand: int = 0,
        verbose: bool = True,
        *args,
        **kwargs,
    ):
        """Make predictions from the model.

        Args:
            return_sites:
                the list of numpyro site names to return values from
            data_single:
                data to condition on which stays the same across all samples
            data_sample:
                data to condition on which is specified for each sample
            num_samples:
                the number of i.i.d. samples to draw
            num_obs:
                the dimension of each of those samples
            rand:
                random seed
            verbose:
                if True, the code prints debugging outputs related to the shapes
                within the numpyro model
            args:
                Not used
            kwargs:
                Not used

        Returns: dictionary of predicted samples
        """
        # Check numbers of samples
        if data_sample is None and num_samples is None:
            raise RuntimeError(
                "Please specify either data_sample or num_samples so that the number of samples desired is known."
            )
        if data_sample is not None:
            num_ls = [v.shape[0] for v in data_sample.values()]
            num_set = set(num_ls)
            if len(num_set) > 1:  # There are multiple lengths of samples
                raise RuntimeError(
                    "Non-matching numbers of requested samples implied by data_sample."
                )
            else:
                num_samples = num_set.pop()

        # Condition the model directly with conditions that apply for all
        # predictive samples
        pred_mod = self.get_model(n_obs=num_obs, *args, **kwargs)
        condition_sites = set()
        if data_single is not None:
            pred_mod = condition(pred_mod, data=data_single)
            condition_sites.update(data_single.keys())

        # Instead of separately conditioning the model, which adds a dimension,
        # simply add the conditioning information to the samples set for
        # conditions which change from sample to sample.
        pred_samples = {}
        if data_sample is not None:
            pred_samples.update(data_sample)
            condition_sites.update(data_sample.keys())

        trained_sites = set(self.samples.keys())
        needed_sites = trained_sites.difference(condition_sites)
        if (
            num_samples < self.avail_fitted_samples
        ):  # Need to sample down the available samples before operating (make sure to make them match!)
            idx = random.choice(
                key=random.PRNGKey(rand),
                a=self.avail_fitted_samples,
                shape=(num_samples,),
                replace=False,
            )
        else:
            idx = slice(None)
        for k in needed_sites:
            pred_samples.update({k: self.samples[k][idx]})

        if verbose:
            self.print_shapes(pred_mod)

        # Run prediction
        predictor = Predictive(
            model=pred_mod,
            posterior_samples=pred_samples,
            return_sites=return_sites,
            parallel=True,
        )
        preds = predictor(random.PRNGKey(rand))

        # If needed, sample extra values from predictions, this saves time since
        # we're not needing to run the model predictively for all of these.
        if num_samples > self.avail_fitted_samples:
            warn(
                f"Predictive Sampling: {num_samples} samples were requested, but only {self.avail_fitted_samples} were sampled during fitting. Sampling with replacement up to the requested number of samples."
            )
            idx = random.choice(
                key=random.PRNGKey(rand),
                a=self.avail_fitted_samples,
                shape=(num_samples,),
                replace=True,
            )
            outs = {k: v[idx] for k, v in preds.items()}
        else:
            outs = preds
        return outs

    def get_samples(self):
        """Get fitted samples from the model."""
        return self.samples

    def set_samples(self, samples: dict):
        """Load fitted samples from the model."""
        self.samples = samples
        avail_ls = [v.shape[0] for v in self.samples.values()]
        avail_set = set(avail_ls)
        if len(avail_set) > 1:  # There are multiple lengths of samples
            raise RuntimeError("Non-matching numbers of samples read from pickle.")
        elif len(avail_set) == 1:
            self.avail_fitted_samples = avail_set.pop()
        else:
            self.avail_fitted_samples = 0

    @abc.abstractmethod
    def get_preds_df(self, preds: dict) -> pd.DataFrame:
        pass

    def print_shapes(self, model=None) -> None:
        """Print shapes of a distribution."""
        if model is None:
            model = self.get_model()
        tr = trace(seed(model, random.PRNGKey(0))).get_trace()
        print(numpyro.util.format_shapes(tr))

    def render_model(self, model=None) -> graphviz.Digraph:
        """Display nodes of distribution with their distribution shapes.

        As of 8/14/2023, this does not work on models with the block effect handler.
        """
        if model is None:
            model = self.get_model()
        graph = numpyro.render_model(model, render_distributions=True)
        return graph
