"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import logging

import geopandas as gpd
import pandas as pd

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.models.manage_charging import _MANAGER_MAP
from megaPLuG.utils.h3 import add_geometries

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
    manager_cls = _MANAGER_MAP[params["charging_manager"]]
    manager = manager_cls(
        dw=dw,
        energy=params["energy_col"],
        dur=params["dwell_dur_col"],
    )
    profs = manager.get_load_profiles(prof_col=params["profile_col"])
    return profs


def report_by_hex(profs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Report results by hex."""
    peaks = profs.groupby(params["id_cols"]["location"]).agg(
        peak_kw=pd.NamedAgg(params["power_col"], "max")
    )
    return peaks


def to_geospatial(df: pd.DataFrame, params: dict) -> gpd.GeoDataFrame:
    """Augment a pandas DataFrame with an H3 id column with the H3 geometries."""
    hexes = add_geometries(df, **params)
    return hexes
