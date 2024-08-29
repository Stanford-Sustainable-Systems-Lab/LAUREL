"""
This is a boilerplate pipeline 'describe_vehicles'
generated using Kedro 0.19.3
"""

import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.params import build_df_from_dict

logger = logging.getLogger(__name__)


def filter_substantial_dwells(dw: DwellSet, params: dict) -> DwellSet:
    """Filter to retain only substantial dwells to be described."""
    dw.data["dwell_hrs"] = (
        dw.data[dw.end] - dw.data[dw.start]
    ).dt.total_seconds() / 3600
    dw.data["long_enough"] = dw.data["dwell_hrs"] > params["thresh_hrs"]
    dw.filter_through("long_enough")
    dw.data = dw.data.drop(
        columns=params["drop_cols"]
    )  # Since these aren't accumulated by filter_through
    return dw


def calc_inter_visit_stats(dw: DwellSet) -> DwellSet:
    """Describe vehicle-location pairs by inter-visit summary statistics."""
    tqdm.pandas()
    # TODO: Consider moving this within DwellSet class and using a Numba for loop
    dw.data = dw.data.groupby(dw.veh, group_keys=False, sort=False).progress_apply(
        calc_inter_visit_times, hex_col=dw.hex, end_col=dw.end, start_col=dw.start
    )

    dw.data["cum_veh_miles"] = dw.data.groupby(dw.veh)[dw.dist].cumsum()
    dw.data["inter_visit_miles"] = dw.data.groupby([dw.veh, dw.hex])[
        "cum_veh_miles"
    ].diff()
    dw.data = dw.data.drop(columns=["cum_veh_miles"])
    return dw


def calc_inter_visit_times(
    grp: pd.DataFrame, hex_col: str, end_col: str, start_col: str
) -> pd.DataFrame:
    """Calculate inter-visit times, assuming that `grp` is from a single vehicle and sorted by time."""
    prev_end_time = grp.groupby(hex_col)[end_col].shift(1)
    grp.loc[:, "inter_visit_hrs"] = (
        grp[start_col] - prev_end_time
    ).dt.total_seconds() / 3600
    return grp


def describe_veh_loc_pairs(dw: DwellSet) -> pd.DataFrame:
    """Describe each vehicle location pair with summary statistics."""
    veh_locs = dw.data.groupby([dw.veh, dw.hex]).agg(
        n_visits=pd.NamedAgg("dwell_hrs", "count"),
        mean_inter_miles=pd.NamedAgg("inter_visit_miles", "mean"),
        med_inter_miles=pd.NamedAgg("inter_visit_miles", "median"),
        max_inter_miles=pd.NamedAgg("inter_visit_miles", "max"),
        mean_inter_times=pd.NamedAgg("inter_visit_hrs", "mean"),
        med_inter_times=pd.NamedAgg("inter_visit_hrs", "median"),
        max_inter_times=pd.NamedAgg("inter_visit_hrs", "max"),
        med_dwell_hrs=pd.NamedAgg("dwell_hrs", "median"),
        tot_dwell_hrs=pd.NamedAgg("dwell_hrs", "sum"),
    )
    veh_locs["dwell_hrs_ratio"] = veh_locs.groupby(dw.veh)["tot_dwell_hrs"].transform(
        lambda s: s / s.sum()
    )
    veh_locs["visits_ratio"] = veh_locs.groupby(dw.veh)["n_visits"].transform(
        lambda s: s / s.sum()
    )
    return veh_locs


def cluster_veh_loc_pairs(veh_locs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Cluster vehicle-location pairs to uncover latent groups."""
    # Prepare for clustering by standardizing variables
    clusterable = deepcopy(veh_locs.dropna(axis=0))
    clusterable = clusterable.drop(columns=params["drop_cols"])
    spars = params["sample"]
    if spars["active"]:
        clusterable = clusterable.sample(n=spars["n"], random_state=spars["seed"])

    ratio_cols = [col for col in clusterable.columns if not col.endswith("_ratio")]
    for col in ratio_cols:
        clusterable.loc[:, col] = clusterable[col] + 1
    scaler = StandardScaler()
    X = scaler.fit_transform(np.log10(clusterable.values))

    # Perform clustering
    n_obs = X.shape[0]
    logger.info(f"Beginning clustering on {n_obs} observations")
    min_clust_size = int(n_obs / params["min_cluster_size_denom"])
    clusterer = HDBSCAN(min_cluster_size=min_clust_size)
    clcol = params["cluster_col"]
    clusterable[clcol] = pd.Categorical(clusterer.fit_predict(X))

    # Merge results back on to original dataframe
    veh_locs = veh_locs.merge(clusterable.loc[:, [clcol]], on=params["merge_cols"])
    return veh_locs


def label_veh_loc_pairs(veh_locs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Set the vehicles' important locations, like home base, if one exists."""
    corpars = params["cluster_location_corresp"]
    col_ls = list(corpars["cols"].values())
    cl_loc_cor = build_df_from_dict(d=corpars["vals"], cols=col_ls)
    cl_loc_cor[corpars["cols"]["location"]] = pd.Categorical(
        cl_loc_cor[corpars["cols"]["location"]]
    )
    veh_locs = veh_locs.merge(cl_loc_cor, how="left", on=corpars["cols"]["cluster"])
    return veh_locs
