"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.19.1
"""
import matplotlib
import pandas as pd

from megaPLuG.models.charging_pgm import DistanceTimePGM


def train_pgm_full(
    train: pd.DataFrame, struct_opts: dict, train_opts: dict
) -> DistanceTimePGM:
    """Train the full PGM."""
    raise NotImplementedError()
    # cond_dict = {
    #     "voc": train["voc_cat_fac"].values,
    #     "far": train["distance_total_mi"].values,
    #     "times": train.loc[:, ["hrs_offshift", "hrs_active", "hrs_break"]].values
    #     / struct_opts["total_hrs"],
    #     "day_shift": train["end_hrs"].values / struct_opts["total_hrs"],
    # }
    # model = DistanceTimePGM(params=struct_opts)
    # model.fit(
    #     data=cond_dict,
    #     num_warmup=train_opts["num_warmup"],
    #     num_samples=train_opts["num_samples"],
    # )
    # return model


def get_posterior_predictive(
    model: DistanceTimePGM,
    params_pred: dict,
) -> pd.DataFrame:
    raise NotImplementedError()
    # model.params["output_schedule"] = True
    # preds = model.predict(
    #     return_sites=["voc", "far", "times", "day_shift", "sched"],
    #     num_samples=params_pred["n_samples"],
    #     num_obs=params_pred["n_vehs_per_fleet"],
    #     verbose=params_pred["verbose"],
    # )
    # # TODO: Remove the "to dataframe" part of this and instead return an InferenceData, which can get combined with training data later for evaluation purposes.
    # sample_df = model.get_preds_df(preds=preds)
    # return sample_df


def eval_single_vars(model: DistanceTimePGM) -> matplotlib.figure.Figure:
    raise NotImplementedError()
    # idata = az.from_numpyro(model.mcmc)
    # axes = az.plot_density(
    #     [idata["prior"], idata["posterior"]],
    #     data_labels=["Observed", "Posterior"],
    #     var_names=["far"],
    #     shade=0.2,
    # )
    # fig = axes.flatten()[0].get_figure()
    # fig.suptitle("94% High Density Intervals for Theta")
    # return fig
