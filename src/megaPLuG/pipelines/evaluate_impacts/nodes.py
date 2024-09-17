"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import logging

import geopandas as gpd
import pandas as pd

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.h3 import cells_to_polygons

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
    dw.seq_names = params["seq_names"]
    hex_kw_cols = [f"{seqn}_{params['event_col']}" for seqn in dw.seq_names]
    dw.data[hex_kw_cols[0]] = (
        dw.data[params["energy_col"]] / dw.data[params["dwell_dur_col"]]
    )
    dw.data[hex_kw_cols[1]] = -dw.data[hex_kw_cols[0]]
    dw.data = dw.data.dropna(subset=hex_kw_cols)
    events = dw.to_events()
    # Sort by hexagon and time
    events = DwellSet._sort_by_grp_time(
        df=events,
        grp_col=dw.hex,
        time_col=DwellSet._get_seq_name_tail(dw.seq_names[0], dw.start),
        drop_cur_idx=True,
    )
    profs = events.groupby(dw.hex)[params["event_col"]].cumsum()
    profs = profs.to_frame()
    return profs


def report_by_hex(profs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Report results by hex."""
    peaks = profs.groupby(params["id_cols"]["location"]).agg(
        peak_kw=pd.NamedAgg(params["event_col"], "max")
    )
    return peaks


def add_geometries(df: pd.DataFrame, params: dict) -> gpd.GeoDataFrame:
    """Augment a pandas DataFrame with an H3 id column with the H3 geometries."""
    hcol = params["hex_col"]
    if hcol in df.columns:
        id_ser = df[hcol]
    elif hcol in df.index.names:
        id_ser = df.index.get_level_values(hcol).to_series()
    else:
        raise RuntimeError(f"'{hcol}' not found in DataFrame columns or index.")

    hexes = gpd.GeoDataFrame(df, geometry=cells_to_polygons(id_ser))
    return hexes
