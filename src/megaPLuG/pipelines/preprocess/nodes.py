"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

import logging
import re
from io import StringIO

import dask.dataframe as dd
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import OneHotEncoder

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.h3 import str_to_h3
from megaPLuG.utils.params import build_df_from_dict
from megaPLuG.utils.time import total_hours

logger = logging.getLogger(__name__)


def format_trips_columns(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Preprocess trips data columns."""
    trips = trips.categorize(params["category_columns"])

    for col in params["time_columns"]:
        # WARNING: This line somehow converts vehicle_id to a float64 from an int64,
        # to fix this, I'm categorizing the vehicle_id column first.
        trips[col] = dd.to_datetime(trips[col], utc=True)

    for col in params["h3_columns"]:
        trips[col] = trips[col].map_partitions(str_to_h3, meta=(col, "int"))

    trips = trips.rename(columns={v: k for k, v in params["col_renamer"].items()})
    trips[params["veh_id_col"]] = trips[params["veh_id_col"]].cat.codes.astype(np.int64)

    if params["persist"]:
        trips = trips.persist()

    return trips


def strip_vehicle_attrs(
    trips: dd.DataFrame, params: dict
) -> tuple[dd.DataFrame, pd.DataFrame]:
    """Get vehicle-specific attributes which stay constant."""
    trips = trips.rename(columns={v: k for k, v in params["col_renamer"].items()})

    n_trips_by_veh = trips[params["veh_id_col"]].value_counts().compute()
    drop_idx = n_trips_by_veh.loc[n_trips_by_veh < params["min_trips_per_veh"]].index

    veh_cols = [params["veh_id_col"]] + params["veh_attr_cols"]
    vehs = trips.loc[:, veh_cols].drop_duplicates().compute()
    vehs = vehs.set_index(params["veh_id_col"]).sort_index()
    vehs = vehs.drop(index=drop_idx)

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


def build_vius_scaling_totals(vius: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Build a scaling factor dependent on home base state and weight class."""
    corresp_hb = build_df_from_dict(
        d=params["home_base_corresp"]["values"],
        id_cols=params["home_base_corresp"]["id_columns"],
        value_col="home_base_code",
    )
    corresp_cab = build_df_from_dict(
        d=params["cab_type_corresp"]["values"],
        id_cols=params["cab_type_corresp"]["id_columns"],
        value_col="cab_type_code",
    )
    # TODO: Then impute Day Cab and Sleeper Cab for unreported using reported ratio
    scaler = vius.rename(columns={v: k for k, v in params["col_renamer"].items()})
    scaler = scaler.merge(corresp_hb, how="left", on=params["home_source_col"])
    scaler = scaler.merge(corresp_cab, how="left", on=params["cab_source_col"])

    # Set up selection series
    enc = OneHotEncoder(sparse_output=False)
    ohot = enc.fit_transform(scaler.loc[:, ["home_base_code"]])
    ohot = pd.DataFrame(ohot, columns=enc.categories_[0], dtype=bool)
    is_reported = ~ohot["Not Reported"] & ~ohot["Not In Use"]

    # Set up grouping series
    scaler.loc[ohot["Home Base in Register State"], params["id_cols"]["region"]] = (
        scaler.loc[ohot["Home Base in Register State"], "reg_state"]
    )
    scaler.loc[ohot["No Home Base"], params["id_cols"]["region"]] = scaler.loc[
        ohot["No Home Base"], "home_base_code"
    ]

    # Calculate weight adjustments
    weights = scaler[params["totals_col"]]
    p_home_base_known_g_has_home_base = (
        ohot["Home Base in Register State"] * weights
    ).sum() / (
        (ohot["Home Base in Register State"] | ohot["Home Base not in Register State"])
        * weights
    ).sum()
    p_is_reported = (is_reported * weights).sum() / (
        ~ohot["Not In Use"] * weights
    ).sum()

    scaler.loc[is_reported, "reported_mult"] = 1 / p_is_reported
    scaler["reported_mult"] = scaler["reported_mult"].fillna(1.0)
    scaler.loc[ohot["Home Base in Register State"], "specific_mult"] = (
        1 / p_home_base_known_g_has_home_base
    )
    scaler["specific_mult"] = scaler["specific_mult"].fillna(1.0)

    drop_idx = scaler.loc[
        ~is_reported | ohot["Home Base not in Register State"] | ohot["Not In Use"]
    ].index
    reduced = scaler.drop(drop_idx)
    reduced[params["totals_col"]] = (
        reduced[params["totals_col"]]
        * reduced["reported_mult"]
        * reduced["specific_mult"]
    )

    orig_wgt = scaler.loc[~ohot["Not In Use"], params["totals_col"]].sum()
    new_wgt = reduced[params["totals_col"]].sum()

    if not np.isclose(orig_wgt, new_wgt):
        raise RuntimeError(
            "Redistributed total weight does not match original total weight."
        )

    id_cols = list(params["id_cols"].values())
    totals = scaler.groupby(id_cols)[params["totals_col"]].sum()
    totals = totals.reset_index()
    return totals
