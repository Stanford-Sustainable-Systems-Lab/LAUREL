"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import logging

import geopandas as gpd
import pandas as pd

from megaPLuG.utils.h3 import h3_to_poly

logger = logging.getLogger(__name__)


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
