"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import logging

import geopandas as gpd
import pandas as pd
from timezonefinder import TimezoneFinder

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.models.manage_charging import _MANAGER_MAP
from megaPLuG.utils.h3 import add_geometries
from megaPLuG.utils.time import calc_local_time_attrs, get_timezone_from_hex

logger = logging.getLogger(__name__)


def summarize_vehicles(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Summarize the results for each vehicle."""
    dw.data["is_death"] = dw.data[params["dead_energy_col"]] < 0
    n_deaths = dw.data.groupby(dw.veh)["is_death"].sum()
    n_deaths.name = "n_deaths"
    vehs = vehs.merge(n_deaths, how="inner", on=dw.veh)

    logger.info("Deaths per vehicle:")
    logger.info(n_deaths.describe())
    return vehs


def get_load_profiles(dw: DwellSet, params: dict) -> pd.DataFrame:
    """Convert vehicle dwells to hexagon load profiles.

    Depending on the charging management algorithm, the transformation to events from
    dwells may occur before or after the power levels. For systems which treat each
    dwell independently, then sending to events after calculating power makes sense. In
    contrast, for systems which consider dwells together, then sending to events before
    calculating power makes sense.
    """
    # Drop dwells with NaN charging energy, which probably resulted from vehicle deaths
    dw.data = dw.data.dropna(subset=params["input_cols"]["energy"])

    # Manage charging energy into power
    manager_cls = _MANAGER_MAP[params["charging_manager"]]
    manager = manager_cls(dw=dw, **params["input_cols"])
    profs = manager.get_load_profiles(prof_col=params["profile_col"])
    return profs


def report_by_hex(profs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Report results by hex."""
    orig_idx = profs.index.names
    profs = profs.reset_index()
    id_cols = params["id_cols"]
    loc_col = id_cols["location"]
    time_col = id_cols["time"]

    logger.info("Finding peaks")
    max_idx = profs.groupby(loc_col)[params["power_col"]].idxmax()
    peaks = profs.loc[max_idx]

    logger.info("Getting time zones")
    tf = TimezoneFinder(in_memory=True)
    peaks[params["timezone_col"]] = peaks[loc_col].transform(
        get_timezone_from_hex, tf=tf
    )

    logger.info("Calculating local time attributes")
    peaks = calc_local_time_attrs(
        df=peaks,
        time_cols=time_col,
        attrs=params["local_time_attrs"],
        tz_col=params["timezone_col"],
    )

    peaks = peaks.set_index(orig_idx)
    peaks = peaks.drop(columns=params["drop_cols"])
    return peaks


def to_geospatial(df: pd.DataFrame, params: dict) -> gpd.GeoDataFrame:
    """Augment a pandas DataFrame with an H3 id column with the H3 geometries."""
    hexes = add_geometries(df, **params)
    return hexes
