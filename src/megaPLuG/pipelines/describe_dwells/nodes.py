import logging
from copy import deepcopy

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from megaPLuG.models.dwell_sets import CumAggFunc, DwellSet
from megaPLuG.utils.h3 import str_to_h3
from megaPLuG.utils.time import total_hours

logger = logging.getLogger(__name__)


def format_trips_columns(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Preprocess trips data columns."""
    trips = trips.categorize(params["category_columns"])

    for col in params["time_columns"]:
        # WARNING: This line somehow converts vehicle_id to a float64 from an int64,
        # to fix this, I'm categorizing the vehicle_id column first.
        trips[col] = dd.to_datetime(trips[col], utc=True)

    for col in params["h3_columns"]:
        trips[col] = trips[col].map_partitions(str_to_h3, meta=(col, "int"))

    trips = trips.rename(columns={v: k for k, v in params["col_renamer"].items()})
    trips[params["veh_id_col"]] = trips[params["veh_id_col"]].cat.codes.astype(np.int64)

    if params["persist"]:
        trips = trips.persist()

    return trips


def calc_derived_trip_cols(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Calculate derived variables which are needed for events."""
    trips["trip_hrs"] = total_hours(
        trips[params["time_cols"]["trip_end"]]
        - trips[params["time_cols"]["trip_start"]]
    )

    if params["persist"]:
        trips = trips.persist()
    return trips


def create_dwells(trips: dd.DataFrame, params: dict, client: Client) -> DwellSet:
    """Create dwell data from trips data."""
    if params["debug_subsample"]["active"]:
        trips = trips.loc[0 : params["debug_subsample"]["n"]]

    if params["load_into_memory"]:
        logger.info("Loading dataset into memory")
        trips = trips.compute()

    logger.info("Converting to dwells from trips.")
    drop_cols = list(set(params["drop_cols"]).intersection(trips.columns))
    trips = trips.drop(columns=drop_cols)
    trips = trips.rename(columns={v: k for k, v in params["col_renamer"].items()})
    colnames = params["from_trips_cols"]
    dw = DwellSet.from_trips(
        trips=trips,
        verify_sorting=params["verify_sorting"],
        **colnames,
        **params["set_index_kwargs"],
    )
    return dw


def coalesce_interrupted_dwells(dw: DwellSet, params: dict) -> DwellSet:
    """Coalesce dwells interrupted by trips with identical origin and destination which
    are short in time and distance.

    This algorithm assumes that the distances and durations for these short trips are
    negligible for the purposes of a charging model, so they are dropped instead of
    being accumulated. The algorithm could be modified easily to have them included.

    If the distances and times do not need to be accumulated, then an alternative
    implementation would simply drop short trips from the trips dataset before dwells
    were created.
    """
    prev_col = f"{dw.hex}_prev"
    mask_col = "is_not_short_circle"

    max_dist = params["max_short_dist_miles"]
    max_dur = params["max_short_dur_hrs"]

    logger.info("Setting shift id column")
    shift_kwargs = {"fill_value": 0}
    if dw.is_dask:
        shift_kwargs["meta"] = ("x", dw.data[dw.hex].dtype)
    dw.data[prev_col] = dw.data.groupby(dw.veh)[dw.hex].shift(1, **shift_kwargs)
    dw.data[mask_col] = ~(
        (dw.data[prev_col] == dw.data[dw.hex])
        & (dw.data[dw.trip_dist] < max_dist)
        & (dw.data[dw.trip_dur] < max_dur)
    )
    dw.data = dw.data.drop(columns=[prev_col])

    # Comments suggest some modifications to apply if distance and duration of dropped
    # trips should be retained.
    logger.info("Accumulating columns across coalescing dwells")
    reset_orig_col = f"{dw.reset}_original"
    dw.data[reset_orig_col] = dw.data[dw.reset].copy()
    dw.data[dw.reset] = (
        False  # Necessary to avoid the special reset-handling in accum_masked
    )
    accum_cols = [reset_orig_col, dw.end]  # "trip_miles", "trip_hrs", "dwell_hrs"
    agg_funcs = [
        CumAggFunc.MAX,
        CumAggFunc.MAX,
    ]  # CumAggFunc.SUM, CumAggFunc.SUM, CumAggFunc.SUM

    dw.data[dw.end] = dw.data[dw.end].astype(np.int64)
    dw.accum_masked(
        keep_mask_col=mask_col,
        accum_cols=accum_cols,
        reverse=True,
        agg_func=agg_funcs,
        write_all=False,
        inplace=True,
    )
    mod_col = f"{dw.end}_{mask_col}"

    mod_times = dw.data[mod_col].astype("datetime64[ns]")
    orig_hex_dtype = dw.data[dw.hex].dtype
    if dw.is_dask:
        dw.data[mod_col] = dd.to_datetime(mod_times, utc=True)
    else:
        dw.data[mod_col] = pd.to_datetime(mod_times, utc=True)

    # Needed because in datetime conversion, hex_id gets float-ified, somehow
    dw.data[dw.hex] = dw.data[dw.hex].astype(orig_hex_dtype)

    logger.info("Dropping coalesced dwells")
    dw.data[mask_col] = dw.data[mask_col].astype("boolean")
    dw.data[mask_col] = dw.data[mask_col].replace(False, pd.NA)
    dw.data = dw.data.dropna(subset=mask_col)
    dw.data[mask_col] = dw.data[mask_col].astype(bool)
    drop_cols = [mask_col] + accum_cols + [reset_orig_col, dw.reset]
    dw.data = dw.data.drop(columns=drop_cols)
    renamer = {f"{old}_{mask_col}": old for old in accum_cols}
    dw.data = dw.data.rename(columns=renamer)

    # Set reset back to its original column name and data type
    renamer = {reset_orig_col: dw.reset}
    dw.data = dw.data.rename(columns=renamer)
    dw.data[dw.reset] = dw.data[dw.reset].astype(bool)
    return dw


def calc_rolling_dwell_ratios(dw: DwellSet, params: dict) -> DwellSet:
    """Calculate the maximum rolling dwell ratios for each vehicle-location pair."""
    dw.data["dur_hrs_col"] = total_hours(dw.data[dw.end] - dw.data[dw.start])

    out_col = params["output_ratio_col"]

    kws = {
        "val_col": "dur_hrs_col",
        "veh_col": dw.veh,
        "loc_col": dw.hex,
        "time_col": dw.start,
        "out_col": out_col,
    }
    kws.update(params["rolling_kwargs"])
    if dw.is_dask:
        meta = dd.utils.make_meta(dw.data)
        meta[out_col] = np.array([], dtype=np.float64)
        kws["meta"] = meta
        dw.data = dw.data.map_partitions(func=_calc_rolling_dwell_ratios_part, **kws)
    else:
        dw.data = _calc_rolling_dwell_ratios_part(dw.data, **kws)

    dw.data = dw.data.drop(columns=["dur_hrs_col"])
    return dw


def _calc_rolling_dwell_ratios_part(
    part: pd.DataFrame,
    out_col: str,
    val_col: str,
    veh_col: str,
    loc_col: str,
    time_col: str,
    **roll_kwargs: dict,
) -> pd.DataFrame:
    part[out_col] = part.groupby(veh_col, sort=False, group_keys=False).apply(
        func=_calc_rolling_dwell_ratios_one_veh,
        val_col=val_col,
        loc_col=loc_col,
        time_col=time_col,
        **roll_kwargs,
    )
    return part


def _calc_rolling_dwell_ratios_one_veh(
    dwells: pd.DataFrame, val_col: str, loc_col: str, time_col: str, **roll_kwargs: dict
) -> pd.Series:
    """Calculate the maximum rolling dwell ratios for each location for a single vehicle."""
    roller = dwells.loc[:, [time_col, loc_col, val_col]]
    roller = roller.set_index(time_col)
    loc_specific = roller.groupby(loc_col)[val_col].rolling(**roll_kwargs).sum()
    loc_specific.name = "loc_specific"
    loc_specific = loc_specific.reset_index(loc_col)
    overall = roller[val_col].rolling(**roll_kwargs).sum()
    overall.name = "overall"

    ratior = pd.concat([loc_specific, overall], axis=1, ignore_index=False)
    ratior["ratio"] = ratior["loc_specific"] / ratior["overall"]

    loc_max_ratios = ratior.groupby(loc_col)["ratio"].max()
    out = dwells[loc_col].map(loc_max_ratios)

    return out


def map_location_groups(
    dw: DwellSet, hex_corresp: pd.DataFrame, params: dict
) -> DwellSet:
    """Map location groups onto the DwellSet."""
    grp_col = params["location_group_col"]
    map_ser = hex_corresp[grp_col]

    if dw.is_dask:
        meta = (grp_col, map_ser.dtype)
        dw.data[grp_col] = dw.data[dw.hex].map(map_ser, meta=meta)
    else:
        dw.data[grp_col] = dw.data[dw.hex].map(map_ser)

    mpars = params["missing_values"]
    if mpars["fill_missing"]:
        dw.data[grp_col] = dw.data[grp_col].fillna(mpars["fill_value"])
    return dw


### vvv Functions below here not used as of 10/20/2025, but preserved for later use vvv


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
        max_dwell_hrs_roll_ratio=pd.NamedAgg("dwell_hrs_roll_ratio", "max"),
    )
    veh_locs["dwell_hrs_ratio"] = veh_locs.groupby(dw.veh, sort=False)[
        "tot_dwell_hrs"
    ].transform(lambda s: s / s.sum())
    veh_locs["visits_ratio"] = veh_locs.groupby(dw.veh, sort=False)[
        "n_visits"
    ].transform(lambda s: s / s.sum())
    return veh_locs


def filter_substantial_dwells(dw: DwellSet, params: dict) -> DwellSet:
    """Filter to retain only substantial dwells to be described."""
    dw.data["dwell_hrs"] = total_hours(dw.data[dw.end] - dw.data[dw.start])
    dw.data["long_enough"] = dw.data["dwell_hrs"] > params["thresh_hrs"]
    accum_cols = [dw.trip_dist, dw.trip_dur, dw.reset]
    dw.accum_masked("long_enough", accum_cols=accum_cols, inplace=True)
    dw.data = dw.data.drop(columns=accum_cols)
    dw.data = dw.data.rename(columns={f"{col}_long_enough": col for col in accum_cols})
    dw.drop_masked(keep_mask_col="long_enough", inplace=True)
    return dw


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
