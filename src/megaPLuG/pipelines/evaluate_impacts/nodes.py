"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.models.group_times import DateGrouper, HourOfWeekdayGrouper
from megaPLuG.models.summarize import EventExpander, NonzeroGroupedSummarizer
from megaPLuG.utils.h3 import cells_to_region_polygons
from megaPLuG.utils.time import (
    calc_local_time,
    calc_time_attrs,
    total_hours,
)

logger = logging.getLogger(__name__)


def summarize_vehicles(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Summarize the results for each vehicle."""
    dw.data["is_death"] = dw.data[params["dead_energy_col"]] < 0
    n_deaths = dw.data.groupby(dw.veh, sort=False)["is_death"].sum()
    n_deaths.name = "n_deaths"
    vehs = vehs.merge(n_deaths, how="inner", on=dw.veh)

    logger.info("Deaths per vehicle:")
    logger.info(n_deaths.describe())
    return vehs


def assign_regions(
    events: pd.DataFrame, hex_regions: pd.DataFrame, pcols: dict
) -> pd.DataFrame:
    """Assign larger regions to the DwellSet based on hexagon ids."""
    orig_idx = events.index.names
    events = events.reset_index()
    grp_cols = list(set(pcols["group_cols"]).intersection(hex_regions.columns))
    reg_cols = [pcols["hex_col"]] + grp_cols + [pcols["timezone_col"]]
    mrg = hex_regions.reset_index().loc[:, reg_cols]
    events = events.merge(mrg, how="left", on=pcols["hex_col"])
    events = events.set_index(orig_idx)
    return events


def assign_vehicle_metadata(
    events: pd.DataFrame, vehs: pd.DataFrame, pcols: dict
) -> pd.DataFrame:
    """Assign larger regions to the DwellSet based on hexagon ids."""
    orig_idx = events.index.names
    events = events.reset_index()
    grp_cols = list(set(pcols["group_cols"]).intersection(vehs.columns))
    if len(grp_cols) > 0:
        veh_cols = [pcols["veh_col"]] + grp_cols
        mrg = vehs.reset_index().loc[:, veh_cols]
        events = events.merge(mrg, how="left", on=pcols["veh_col"])
    events = events.set_index(orig_idx)
    return events


def get_load_profiles(events: pd.DataFrame, params: dict, pcols: dict) -> pd.DataFrame:
    """Convert vehicle charging events to load profiles by group."""
    logger.info("Sorting events by group and time")
    events = events.reset_index()
    if params["drop_null_groups"]:
        events = events.dropna(subset=pcols["group_cols"])
    events = events.sort_values(pcols["group_cols"] + [pcols["time_col"]])

    logger.info("Calculating load profiles by accumulating events.")
    event_grp = events.groupby(pcols["group_cols"], sort=False, observed=True)
    # Note: The vehicle and hexagon ids are rendered uninterpretable by the cumsum
    events[pcols["profile_col"]] = event_grp[params["power_col"]].cumsum()
    events[pcols["duration_col"]] = event_grp[pcols["time_col"]].transform(
        lambda ser: ser.shift(-1) - ser
    )
    profs = events.drop(columns=[params["power_col"]])
    profs = profs.set_index(pcols["group_cols"] + [pcols["time_col"]])
    return profs


def report_by_region_peaks(
    profs: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Report results by hex."""
    orig_idx = profs.index.names
    profs = profs.reset_index()

    logger.info("Finding peaks")
    max_idx = profs.groupby(pcols["group_cols"], sort=False, observed=True)[
        pcols["profile_col"]
    ].idxmax()
    peaks = profs.loc[max_idx]

    logger.info("Calculating local time attributes")
    peaks = calc_local_time(
        df=peaks,
        time_cols=pcols["time_col"],
        local_cols=pcols["time_col"] + "_local",
        tz_col=pcols["timezone_col"],
    )
    peaks = peaks.set_index(orig_idx)
    return peaks


def report_by_region_quantiles(
    profs: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Report quantile summaries by region and time grouping."""
    logger.info("Remove all observations with unknown duration or zero power.")
    # First drop the observations with no duration or zero power
    nonzero = profs.dropna(subset=[pcols["duration_col"]])
    nonzero = nonzero.reset_index()
    drop_idx = nonzero.loc[nonzero[pcols["profile_col"]] == 0].index
    nonzero = nonzero.drop(index=drop_idx)

    logger.info("Expand events to cover all groups across their duration")
    grp_tz_cols = pcols["group_cols"] + [pcols["timezone_col"]]
    expander = EventExpander(
        time_col=pcols["time_col"],
        dur_col=pcols["duration_col"],
        value_col=pcols["profile_col"],
        group_cols=grp_tz_cols,
        freq=params["freq"],
    )
    nonzero_exp = expander.expand_events(nonzero)

    logger.info("Group events")
    grouper = grp_tz_cols + [pd.Grouper(key=pcols["time_col"], freq=params["freq"])]
    grped_nonzero = nonzero_exp.groupby(grouper)[pcols["profile_col"]].max()
    grped_nonzero = grped_nonzero.reset_index()

    all_times = profs.index.get_level_values(pcols["time_col"])
    grouper = HourOfWeekdayGrouper(
        time_col=pcols["time_col"],
        tz_col=pcols["timezone_col"],
        start_time=all_times.min(),
        end_time=all_times.max(),
        possible_tzs=profs[pcols["timezone_col"]].unique(),
    )
    grped_nonzero = grouper.add_group_classes(grped_nonzero)
    group_counts_tz = grouper.get_possible_obs_counts().reset_index()
    grp_merge_cols = [pcols["timezone_col"]] + grouper.time_group_cols
    grped_nonzero = grped_nonzero.merge(group_counts_tz, how="left", on=grp_merge_cols)

    logger.info("Calculate quantiles")
    summ_cols = pcols["group_cols"] + grouper.time_group_cols
    summer = NonzeroGroupedSummarizer(
        group_cols=summ_cols,
        quantiles=np.array(params["quantiles"]),
    )
    quantiles = summer.summarize(
        events=grped_nonzero,
        value_col=pcols["profile_col"],
        possible_count_col="possible_count",
    )
    return quantiles


def report_by_region_capacity_consumed(
    profs: pd.DataFrame, baseload: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Report quantile summaries by region and time grouping."""
    logger.info("Remove all observations with unknown duration or zero power.")
    # First drop the observations with no duration or zero power
    nonzero = profs.dropna(subset=[pcols["duration_col"]])
    nonzero = nonzero.reset_index()
    drop_idx = nonzero.loc[nonzero[pcols["profile_col"]] == 0].index
    nonzero = nonzero.drop(index=drop_idx)

    logger.info("Expand events to cover all groups across their duration")
    grp_tz_cols = pcols["group_cols"] + [pcols["timezone_col"]]
    expander = EventExpander(
        time_col=pcols["time_col"],
        dur_col=pcols["duration_col"],
        value_col=pcols["profile_col"],
        group_cols=grp_tz_cols,
        freq=params["freq"],
    )
    nonzero_exp = expander.expand_events(nonzero)

    logger.info("Group events")
    grouper = grp_tz_cols + [pd.Grouper(key=pcols["time_col"], freq=params["freq"])]
    grped_nonzero = nonzero_exp.groupby(grouper)[pcols["profile_col"]].max()
    grped_nonzero = grped_nonzero.reset_index()

    local_col = pcols["time_col"] + "_local"
    grped_nonzero = calc_local_time(
        df=grped_nonzero,
        time_cols=pcols["time_col"],
        local_cols=local_col,
        tz_col=pcols["timezone_col"],
    )
    grped_nonzero = calc_time_attrs(grped_nonzero, time_col=local_col, attrs="hour")
    # The following line eliminates all profile observations not corresponding to a baseload
    pair_nonzero = grped_nonzero.merge(
        baseload, how="inner", on=params["prof_merge_cols"]
    )

    # Assumes that profile column is measured in kW
    base_cols = params["baseload_cols"]
    pair_nonzero["base_veh_mw"] = (
        pair_nonzero[base_cols["hourly"]]
        + pair_nonzero[pcols["profile_col"]] * params["vehicle_profile_conversion"]
    )

    all_times = profs.index.get_level_values(pcols["time_col"])
    grouper = DateGrouper(
        time_col=pcols["time_col"],
        tz_col=pcols["timezone_col"],
        start_time=all_times.min(),
        end_time=all_times.max(),
        possible_tzs=profs[pcols["timezone_col"]].unique(),
    )
    pair_nonzero = grouper.add_group_classes(pair_nonzero)

    grp_cols = pcols["group_cols"] + grouper.time_group_cols + [pcols["timezone_col"]]
    peak_mod = pair_nonzero.groupby(grp_cols).agg(
        max_base_mw=pd.NamedAgg(base_cols["all_time"], "first"),
        max_veh_mw=pd.NamedAgg(pcols["profile_col"], "max"),
        max_mod_mw=pd.NamedAgg("base_veh_mw", "max"),
        cap_avail_mw=pd.NamedAgg(base_cols["capacity_available"], "first"),
    )
    peak_mod["max_veh_mw"] = (
        peak_mod["max_veh_mw"] * params["vehicle_profile_conversion"]
    )
    peak_mod["diff_mw_abs"] = peak_mod["max_mod_mw"] - peak_mod[base_cols["all_time"]]
    peak_mod["diff_mw_abs"] = np.maximum(peak_mod["diff_mw_abs"], 0)
    peak_mod["diff_mw_pct"] = (
        peak_mod["diff_mw_abs"] / peak_mod[base_cols["capacity_available"]] * 100
    )

    poss_times = grouper.get_possible_obs_counts().reset_index()
    poss_times = poss_times.drop(columns=[grouper.count_col])
    poss_groups = (
        peak_mod.reset_index()
        .loc[:, pcols["group_cols"] + [pcols["timezone_col"]]]
        .drop_duplicates()
    )
    date_frame = poss_groups.merge(poss_times, how="left", on=pcols["timezone_col"])
    full_obs = date_frame.merge(peak_mod, how="left", on=grp_cols)

    # We know that all times with NA vehicle additions actually had zero
    fcols = params["fill_with_zero_cols"]
    full_obs.loc[:, fcols] = full_obs.loc[:, fcols].fillna(0.0)
    full_obs = full_obs.dropna(axis=1)
    full_obs = full_obs.drop(columns=[pcols["timezone_col"]])
    return full_obs


def add_region_geoms(
    results: pd.DataFrame,
    hex_regions: pd.DataFrame,
    params: dict,
) -> gpd.GeoDataFrame:
    """Add region geometries to the reporting by region."""
    reg_polys = cells_to_region_polygons(
        corresp=hex_regions.reset_index(),
        hex_col=params["hex_col"],
        region_col=params["region_col"],
    )
    results = results.merge(reg_polys, on=params["region_col"])
    res_geos = gpd.GeoDataFrame(results, geometry="geometry")

    changed_cols = []
    for col in res_geos.columns:
        if pd.api.types.is_timedelta64_dtype(res_geos[col]):
            res_geos[f"{col}_hrs"] = total_hours(res_geos[col])
            changed_cols.append(col)

    res_geos = res_geos.drop(columns=changed_cols)

    return res_geos
