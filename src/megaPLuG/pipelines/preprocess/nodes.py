"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

import logging

import dask.dataframe as dd
import h3.api.numpy_int as h3
import pandas as pd

from megaPLuG.models.dwell_sets import DwellSet

logger = logging.getLogger(__name__)


def format_trips_columns(trips, params):
    """Preprocess trips data columns."""
    trips = trips.categorize(params["category_columns"])

    for col in params["time_columns"]:
        trips[col] = dd.to_datetime(trips[col], utc=True)

    for col in params["h3_columns"]:
        trips[col] = trips[col].map_partitions(str_to_h3, meta=(col, "int"))

    if params["persist"]:
        trips = trips.persist()

    return trips


def str_to_h3(s: pd.Series) -> pd.Series:
    return s.transform(h3.str_to_int)


def create_dwells(trips, params):
    """Create dwell data from trips data."""
    if params["debug_subsample"]["active"]:
        trips = trips.loc[0 : params["debug_subsample"]["n"]]

    if params["load_into_memory"]:
        logger.info("Loading dataset into memory")
        trips = trips.compute()

    logger.info("Converting to dwells from trips.")
    trips = trips.drop(columns=params["drop_cols"])
    trips = trips.rename(columns={v: k for k, v in params["col_renamer"].items()})
    colnames = params["from_trips_cols"]
    dw = DwellSet.from_trips(
        trips=trips,
        veh=colnames["veh"],
        hex=colnames["hex"],
        start_trip=colnames["start_trip"],
        end_trip=colnames["end_trip"],
        dist=colnames["dist"],
    )

    if params["persist"]:
        dw.data = dw.data.persist()

    return dw.data


def clean_vius(vius):
    """Remove unnecessary VIUS columns and calculate new ones, if necessary."""
    raise NotImplementedError()
