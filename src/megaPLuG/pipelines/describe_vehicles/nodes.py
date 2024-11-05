"""
This is a boilerplate pipeline 'describe_vehicles'
generated using Kedro 0.19.3
"""

import logging
from copy import deepcopy

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.data import merge_on_int_cols
from megaPLuG.utils.h3 import add_geometries
from megaPLuG.utils.params import build_df_from_dict
from megaPLuG.utils.time import total_hours

logger = logging.getLogger(__name__)


def filter_substantial_dwells(dw: DwellSet, params: dict) -> DwellSet:
    """Filter to retain only substantial dwells to be described."""
    dw.data["dwell_hrs"] = total_hours(dw.data[dw.end] - dw.data[dw.start])
    dw.data["long_enough"] = dw.data["dwell_hrs"] > params["thresh_hrs"]
    dw.accum_masked("long_enough", inplace=True)
    dw.data = dw.data.drop(
        columns=params["drop_cols"]
    )  # Since these aren't accumulated by accum_masked
    return dw


def calc_inter_visit_stats(dw: DwellSet) -> DwellSet:
    """Describe vehicle-location pairs by inter-visit summary statistics."""
    tqdm.pandas()
    # TODO: Consider moving this within DwellSet class and using a Numba for loop
    dw.data = dw.data.groupby(dw.veh, group_keys=False, sort=False).progress_apply(
        calc_inter_visit_times, hex_col=dw.hex, end_col=dw.end, start_col=dw.start
    )

    dw.data["cum_veh_miles"] = dw.data.groupby(dw.veh, sort=False)[
        dw.trip_dist
    ].cumsum()
    dw.data["inter_visit_miles"] = dw.data.groupby([dw.veh, dw.hex], sort=False)[
        "cum_veh_miles"
    ].diff()
    dw.data = dw.data.drop(columns=["cum_veh_miles"])
    return dw


def calc_inter_visit_times(
    grp: pd.DataFrame, hex_col: str, end_col: str, start_col: str
) -> pd.DataFrame:
    """Calculate inter-visit times, assuming that `grp` is from a single vehicle and sorted by time."""
    prev_end_time = grp.groupby(hex_col, sort=False)[end_col].shift(1)
    grp.loc[:, "inter_visit_hrs"] = total_hours(grp[start_col] - prev_end_time)
    return grp


def describe_veh_loc_pairs(dw: DwellSet) -> pd.DataFrame:
    """Describe each vehicle location pair with summary statistics."""
    veh_locs = dw.data.groupby([dw.veh, dw.hex], sort=False).agg(
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
    veh_locs["dwell_hrs_ratio"] = veh_locs.groupby(dw.veh, sort=False)[
        "tot_dwell_hrs"
    ].transform(lambda s: s / s.sum())
    veh_locs["visits_ratio"] = veh_locs.groupby(dw.veh, sort=False)[
        "n_visits"
    ].transform(lambda s: s / s.sum())
    return veh_locs


def cluster_veh_loc_pairs(veh_locs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Cluster vehicle-location pairs to uncover latent groups."""
    # Prepare for clustering by standardizing variables
    logger.info("Select feature variables")
    clusterable = deepcopy(veh_locs.dropna(axis=0))
    drop_cols = np.setdiff1d(clusterable.columns, params["feature_cols"])
    clusterable = clusterable.drop(columns=drop_cols)

    spars = params["sample"]
    if spars["active"]:
        n = spars["n"]
        logger.info(f"Sample {n} observations")
        clusterable = clusterable.sample(n=n, random_state=spars["seed"])

    logger.info("Log-transform and mean-std scale features.")
    for col in clusterable.columns:
        if not col.endswith("_ratio"):
            # Ratio columns would not benefit from spread reduction of log1p
            clusterable.loc[:, col] = clusterable[col] + 1
        if not col.endswith("_entropy"):
            clusterable.loc[:, col] = np.log10(clusterable[col])
    scaler = StandardScaler()
    clusterable = pd.DataFrame(
        data=scaler.fit_transform(clusterable),
        index=clusterable.index,
        columns=clusterable.columns,
    )

    # Perform clustering
    n_obs = len(clusterable)
    logger.info(f"Beginning clustering on {n_obs} observations")
    min_clust_size = int(n_obs / params["min_cluster_size_denom"])
    clusterer = HDBSCAN(min_cluster_size=min_clust_size)
    clusterer = clusterer.fit(clusterable.values)

    # Merge results back on to original dataframe
    clusters = pd.DataFrame(
        data=clusterer.labels_,
        index=clusterable.index,
        columns=[params["cluster_col"]],
    )
    clusters[params["cluster_col"]] = pd.Categorical(clusters[params["cluster_col"]])
    veh_locs = veh_locs.merge(clusters, left_index=True, right_index=True)
    return veh_locs


def label_veh_loc_pairs(veh_locs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Set the vehicles' important locations, like home base, if one exists."""
    corpars = params["location"]
    cl_loc_cor = build_df_from_dict(
        d=corpars["vals"],
        id_cols=list(corpars["id_cols"].values()),
        value_col="location",
    )
    cl_loc_cor["location"] = pd.Categorical(cl_loc_cor["location"])
    orig_idx = veh_locs.index.names
    veh_locs = veh_locs.reset_index()
    veh_locs = veh_locs.merge(cl_loc_cor, how="left", on=corpars["id_cols"]["cluster"])
    veh_locs = veh_locs.set_index(orig_idx)
    return veh_locs


def classify_vehicles(
    vehs: pd.DataFrame, veh_locs: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Classify vehicles by their route type, home base status, etc."""
    veh_loc_cts = veh_locs.groupby(params["veh_col"], sort=False)[
        params["loc_col"]
    ].value_counts()
    veh_loc_cts = veh_loc_cts.unstack(params["loc_col"])
    veh_loc_cts["has_home_base"] = veh_loc_cts["home_base"] > 0
    vehs = vehs.merge(
        veh_loc_cts.loc[:, ["has_home_base"]], how="left", on=params["veh_col"]
    )
    vehs.loc[vehs["has_home_base"].isna(), "has_home_base"] = False
    vehs["has_home_base"] = vehs["has_home_base"].astype(bool)
    return vehs


def mark_weight_class_group(vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Mark the vehicles with VIUS weight class groups."""
    wgt_corresp = build_df_from_dict(
        params["values"],
        id_cols=params["id_columns"],
        value_col=params["value_col"],
    )
    orig_idx = vehs.index.names
    vehs = vehs.reset_index()
    vehs = vehs.merge(wgt_corresp, how="left", on=params["id_columns"])
    vehs = vehs.set_index(orig_idx)
    return vehs


def mark_location_regions(
    vehs: pd.DataFrame,
    veh_locs: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    params: dict,
    dwell_params: dict,
) -> pd.DataFrame:
    """Mark the region of chosen locations by intersecting location points with region polygons."""
    veh_col = dwell_params["veh"]
    hex_col = dwell_params["hex"]
    loc_col = params["location_col"]
    loc_reg_col = params["location_region_col"]

    veh_locs = veh_locs.reset_index()
    bases = veh_locs.loc[
        ~veh_locs[loc_col].isna(), [veh_col, hex_col, loc_col]
    ].drop_duplicates()
    bases = add_geometries(bases, hex_col=hex_col)
    bases = bases.to_crs(regions.crs)
    mrg = regions.loc[:, [params["region_name_col"], regions.geometry.name]]
    base_regs = bases.sjoin(mrg, how="left")
    base_regs = base_regs.rename(columns={params["region_name_col"]: loc_reg_col})
    base_regs = base_regs.loc[:, [veh_col, loc_reg_col]].drop_duplicates(subset=veh_col)
    base_regs = base_regs.set_index(veh_col)
    vehs = vehs.merge(base_regs, how="left", on=veh_col)
    vehs[loc_reg_col] = vehs[loc_reg_col].fillna(params["na_region_fill"])
    return vehs


def calc_vehicle_scaling_weights(
    vehs: pd.DataFrame,
    scaler: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Calculate vehicle scaling weights based on vehicle count totals.

    Right now, this uses region and weight class.
    """
    if len(params["summaries"]) > 1:
        raise RuntimeError("One summary only is expected.")
    grp_cols = params["group_cols"]

    orig_idx = vehs.index.names
    vehs = vehs.reset_index()
    trip_summ = vehs.groupby(grp_cols, observed=True).agg(params["summaries"])

    trips_name = f"{params['totals_col']}_trips"
    region_name = f"{params['totals_col']}_region"
    summ_col = list(params["summaries"].keys())[0]
    trip_summ = trip_summ.rename(columns={summ_col: trips_name})
    scaler = scaler.merge(trip_summ, how="left", on=grp_cols)
    scaler[params["weight_col"]] = scaler[region_name] / scaler[trips_name]

    mrg = scaler.loc[:, grp_cols + [params["weight_col"]]]
    vehs = vehs.merge(mrg, how="left", on=grp_cols)
    vehs = vehs.set_index(orig_idx)
    return vehs


def mark_locations(dw: DwellSet, veh_locs: pd.DataFrame, params: dict) -> DwellSet:
    """Mark locations-of-interest for each vehicle (e.g. home base)."""
    right = veh_locs.loc[:, params["veh_loc_cols"]]
    merge_cols = [dw.veh, dw.hex]
    if not dw.is_dask:
        dw.data = dw.data.merge(right, how="left", on=merge_cols)
    else:
        dw.data = merge_on_int_cols(left=dw.data, right=right, on=[dw.veh, dw.hex])
    return dw
