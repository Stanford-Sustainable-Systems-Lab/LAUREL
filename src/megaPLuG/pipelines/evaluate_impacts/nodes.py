"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import logging

import geopandas as gpd
import h3.api.numpy_int as h3
import pandas as pd
from shapely.geometry import Polygon

from megaPLuG.models.dwell_sets import DwellSet

logger = logging.getLogger(__name__)


def drop_vehicles(dw: DwellSet, params: dict) -> DwellSet:
    """Drop vehicles which are excluded, for example those which weren't electrifiable"""
    dead_vehs = (
        dw.data.groupby(dw.veh)[params["dead_energy_col"]].last(skipna=False).isna()
    )
    drop_idx = dead_vehs.loc[dead_vehs].index
    dw.data = dw.data.drop(index=drop_idx)

    n_vehs = dead_vehs.shape[0]
    n_dead = dead_vehs.sum()
    n_elect = n_vehs - n_dead
    pct_elect = round(n_elect / n_vehs * 100, 1)
    logger.info(f"Electrifiable vehicles: {n_elect}, {pct_elect}%")
    return dw


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


def h3_to_poly(h: int) -> Polygon:
    bnd = h3.cell_to_boundary(h)
    # h3 outputs geometries in lat-lon format, but the convention in WGS84 is lon-lat
    bnd_flip = [(x, y) for y, x in bnd]
    poly = Polygon(bnd_flip)
    return poly
