"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

import logging
import re
from io import StringIO
from itertools import product

import dask.dataframe as dd
import pandas as pd
import requests

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.h3 import str_to_h3
from megaPLuG.utils.time import total_hours

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
    trips["trip_hrs"] = total_hours(
        trips[params["time_cols"]["trip_end"]]
        - trips[params["time_cols"]["trip_start"]]
    )

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
        trip_dist=colnames["trip_dist"],
        trip_dur=colnames["trip_dur"],
    )

    if not params["load_into_memory"]:
        dw.data = dw.data.persist()

    return dw.data


def get_vius_from_url(url: str, params: dict) -> pd.DataFrame:
    """Get a VIUS dataset from a URL."""
    r = requests.get(url)
    txt = re.sub(r"[\[\]]", "", r.text)
    df = pd.read_csv(StringIO(txt))

    df = df.rename(columns={v: k for k, v in params["col_renamer"].items()})
    df = df.loc[:, list(params["col_renamer"].keys())]

    for col, mult in params["multipliers"].items():
        df[col] = df[col] * mult

    df = df.set_index(params["index_col"]).sort_index()
    return df


def clean_vius_by_home_base_state(vius: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Clean the VIUS VMT by home base state table."""
    vius = vius.drop(index=params["drop_idx_values"])
    orig_idx = vius.index.names
    vius = vius.reset_index()
    for old, new in params["replace_values"].items():
        vius = vius.replace(old, new)
    vius = vius.set_index(orig_idx)
    return vius


def clean_vius_by_weight_class(weights: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Clean the VIUS VMT by weight class table."""
    weights = weights.drop(index=params["drop_idx_values"])
    return weights


def build_vius_scaling_totals(
    regions: pd.DataFrame, weights: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Build a scaling factor dependent on home base state and weight class.

    Note that the VIUS top-level tables do not jointly tabulate along these two
    dimensions, so we assume that they are independent.
    """
    id_cols = params["id_cols"]
    scaler = pd.DataFrame(
        product(regions.index, weights.index), columns=list(id_cols.values())
    )
    tot_col = params["totals_col"]
    new_name = f"weight_class_factor_{params['totals_col']}"
    weights[new_name] = weights[tot_col] / weights[tot_col].sum()
    scaler = scaler.merge(weights, how="left", on=id_cols["weight_class"])
    scaler = scaler.merge(
        regions,
        how="left",
        on=id_cols["region"],
        suffixes=("_weight", "_region"),
    )

    scaler[params["weight_col"]] = scaler[f"{tot_col}_region"] * scaler[new_name]
    return scaler
