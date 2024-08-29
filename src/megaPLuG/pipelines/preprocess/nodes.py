"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

import logging

import dask.dataframe as dd
import pandas as pd

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.h3 import str_to_h3

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


def strip_vehicle_attrs(
    trips: dd.DataFrame, params: dict
) -> tuple[dd.DataFrame, pd.DataFrame]:
    """Get vehicle-specific attributes which stay constant."""
    trips = trips.rename(columns={v: k for k, v in params["col_renamer"].items()})

    vehs = (
        trips.loc[:, [params["veh_id_col"]] + params["veh_attr_cols"]]
        .drop_duplicates()
        .compute()
    )
    vehs = vehs.set_index(params["veh_id_col"]).sort_index()

    trips = trips.drop(columns=params["veh_attr_cols"])
    if params["persist"]:
        trips = trips.persist()

    return (trips, vehs)


def calc_derived_trip_cols(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Calculate derived variables which are needed for events."""
    trips["trip_hrs"] = (
        trips[params["time_cols"]["trip_end"]]
        - trips[params["time_cols"]["trip_start"]]
    ).dt.total_seconds() / 3600
    trips["trip_avg_speed_mph"] = trips["trip_miles"] / trips["trip_hrs"]

    if params["persist"]:
        trips = trips.persist()
    return trips


def create_dwells(trips: dd.DataFrame, params: dict) -> dd.DataFrame | pd.DataFrame:
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
