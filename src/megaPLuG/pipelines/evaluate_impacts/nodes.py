"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import logging

import geopandas as gpd
import pandas as pd

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.h3 import h3_to_poly

logger = logging.getLogger(__name__)


def calc_derived_dwell_cols(dw: DwellSet, params: dict) -> DwellSet:
    """Calculate derived variables which are needed for events."""
    dw.data = dw.data.rename(columns=params["initial_rename"])
    dw.data["dwell_end_veh_kwh"] = (
        dw.data["dwell_start_veh_kwh"] + dw.data["charge_kwh"]
    )
    dw.data["dwell_start_hex_kw_diff"] = (
        dw.data["charge_kwh"] / dw.data["dwell_time_hrs"]
    )
    dw.data["dwell_end_hex_kw_diff"] = -dw.data["dwell_start_hex_kw_diff"]
    return dw


def get_hex_events_from_dwells(dw: DwellSet, params: dict) -> pd.DataFrame:
    """Sort events by hex instead of by vehicle."""
    dw.seq_names = params["seq_names"]
    events = dw.to_hex_profiles()
    return events


def report_by_hex(events: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Report results by hex."""
    profs = events.groupby(params["id_cols"]["location"])[params["event_col"]].cumsum()
    profs = profs.to_frame()
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

    df["polygons"] = id_ser.transform(h3_to_poly)
    hexes = gpd.GeoDataFrame(df, geometry="polygons", crs=params["crs"])
    return hexes
