"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

import logging
from functools import reduce

import dask.dataframe as dd
import geopandas as gpd
import h3.api.numpy_int as h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def format_trips_columns(trips, params):
    """Preprocess trips data columns."""
    trips = trips.categorize(params["category_columns"])

    for col in params["time_columns"]:
        trips[col] = dd.to_datetime(trips[col], utc=True)

    for col in params["h3_columns"]:
        trips[col] = trips[col].map_partitions(str_to_h3, meta=(col, "int"))

    return trips


def str_to_h3(s: pd.Series) -> pd.Series:
    return s.transform(h3.string_to_h3)


def set_trips_index(trips, params):
    """Set index of trips data."""
    if params["debug_subsample"]["active"]:
        trips = trips.loc[0 : params["debug_subsample"]["n"]]
    trips = trips.compute()
    trips = trips.sort_values(by=params["sort_column_order"])
    trips = trips.reset_index(drop=True)
    trips.index.name = params["index_column"]
    return trips


def build_h3_polygons(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Build H3 Scale 8 polygons for all observed locations."""
    hex_ls = [trips[c].unique() for c in params["hex_cols"]]
    ids = reduce(np.intersect1d, hex_ls)
    ids = pd.Series(ids, name="hex_id")

    def h3_to_poly(h: int) -> Polygon:
        return Polygon(h3.h3_to_geo_boundary(h, geo_json=True))

    polys = ids.transform(h3_to_poly)

    geos = gpd.GeoDataFrame(ids, geometry=polys, crs=params["crs"])
    return geos


def clean_vius(vius):
    """Remove unnecessary VIUS columns and calculate new ones, if necessary."""
    raise NotImplementedError()
