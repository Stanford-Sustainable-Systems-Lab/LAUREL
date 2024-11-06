"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.models.group_times import HourOfWeekdayGrouper
from megaPLuG.models.manage_charging import _MANAGER_MAP
from megaPLuG.models.summarize import EventExpander, NonzeroGroupedSummarizer
from megaPLuG.utils.data import ColumnIntegerizer
from megaPLuG.utils.h3 import cells_to_region_polygons
from megaPLuG.utils.time import calc_local_time_attrs, total_hours

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


def assign_regions(dw: DwellSet, hex_regions: pd.DataFrame) -> DwellSet:
    """Assign larger regions to the DwellSet based on hexagin ids."""
    orig_idx = dw.data.index.names
    dw.data = dw.data.reset_index()
    dw.data = dw.data.merge(hex_regions, how="left", on=dw.hex)
    dw.data = dw.data.set_index(orig_idx)
    return dw


def assign_scale_up_factor(
    dw: DwellSet, vehs: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Assign the factor by which each dwell's power will be scaled up."""
    if params["apply_scaling"]:
        mrg = vehs.loc[:, params["veh_cols"]]
        dw.data = dw.data.merge(mrg, how="left", on=dw.veh)
    else:
        dw.data.loc[:, params["veh_cols"]] = 1.0
    return dw


def get_load_profiles(dw: DwellSet, params: dict) -> pd.DataFrame:
    """Convert vehicle dwells to hexagon load profiles.

    Depending on the charging management algorithm, the transformation to events from
    dwells may occur before or after the power levels. For systems which treat each
    dwell independently, then sending to events after calculating power makes sense. In
    contrast, for systems which consider dwells together, then sending to events before
    calculating power makes sense.
    """
    # Drop dwells with NaN charging energy, which probably resulted from vehicle deaths
    icols = params["input_cols"]
    drop_cols = [icols["energy"], icols["region"]]
    dw.data = dw.data.dropna(subset=drop_cols)

    # Manage charging energy into power
    manager_cls = _MANAGER_MAP[params["charging_manager"]]
    manager = manager_cls(dw=dw, **icols)
    profs = manager.get_load_profiles(
        prof_col=params["profile_col"],
        dur_col=params["duration_col"],
    )
    return profs


def report_by_region_peaks(
    profs: pd.DataFrame, hex_regions: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Report results by hex."""
    orig_idx = profs.index.names
    profs = profs.reset_index()
    id_cols = params["id_cols"]
    loc_col = id_cols["location"]
    time_col = id_cols["time"]

    logger.info("Finding peaks")
    max_idx = profs.groupby(loc_col, sort=False)[params["power_col"]].idxmax()
    peaks = profs.loc[max_idx]

    logger.info("Calculating local time attributes")
    hex_regions_merge = hex_regions.loc[:, [loc_col, params["timezone_col"]]]
    hex_regions_merge = hex_regions_merge.drop_duplicates(subset=loc_col, keep="first")
    peaks = peaks.merge(hex_regions_merge, how="left", on=loc_col)
    peaks = calc_local_time_attrs(
        df=peaks,
        time_cols=time_col,
        attrs=params["local_time_attrs"],
        tz_col=params["timezone_col"],
    )

    peaks = peaks.set_index(orig_idx)
    return peaks


def report_by_region_quantiles(
    profs: pd.DataFrame,
    hex_regions: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Report quantile summaries by region and time grouping."""
    pcols = params["columns"]

    logger.info("Remove all observations with unknown duration or zero power.")
    # First drop the observations with no duration or zero power
    nonzero = profs.dropna(subset=[pcols["duration"]])
    nonzero = nonzero.reset_index()
    drop_idx = nonzero.loc[nonzero[pcols["power"]] == 0].index
    nonzero = nonzero.drop(index=drop_idx)

    logger.info("Expand events to cover all groups across their duration")
    region_inter = ColumnIntegerizer(pcols["region"])
    nonzero = region_inter.integerize(nonzero)
    # Then apply expansion
    expander = EventExpander(
        time_col=pcols["time"],
        dur_col=pcols["duration"],
        value_col=pcols["power"],
        group_col=pcols["region"],
        freq=params["freq"],
    )
    nonzero_exp = expander.expand_events(nonzero)
    nonzero_exp = region_inter.deintegerize(nonzero_exp)

    logger.info("Group events")
    grouper = [pcols["region"], pd.Grouper(key=pcols["time"], freq=params["freq"])]
    grped_nonzero = nonzero_exp.groupby(grouper)[pcols["power"]].max()
    grped_nonzero = grped_nonzero.reset_index()

    mrgr = hex_regions.reset_index()[[pcols["region"], "tz"]].drop_duplicates()
    grped_nonzero = grped_nonzero.merge(mrgr, how="left", on=pcols["region"])

    grouper = HourOfWeekdayGrouper(
        time_col=pcols["time"],
        tz_col=pcols["tz"],
    )
    grped_nonzero = grouper.add_group_classes(grped_nonzero)
    group_counts_tz = grouper.get_possible_obs_counts(grped_nonzero)
    group_counts_tz = group_counts_tz.reset_index()
    grp_merge_cols = [pcols["tz"]] + grouper.time_group_cols
    grped_nonzero = grped_nonzero.merge(group_counts_tz, how="left", on=grp_merge_cols)

    logger.info("Calculate quantiles")
    summ_cols = [pcols["region"]] + grouper.time_group_cols
    summer = NonzeroGroupedSummarizer(
        group_cols=summ_cols,
        quantiles=np.array(params["quantiles"]),
    )
    region_inter = ColumnIntegerizer(pcols["region"])
    grped_nonzero = region_inter.integerize(grped_nonzero)
    quantiles = summer.summarize(
        events=grped_nonzero,
        value_col=pcols["power"],
        possible_count_col="possible_count",
    )
    quantiles = region_inter.deintegerize(quantiles)
    return quantiles


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
