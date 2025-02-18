"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

import logging
import re
from io import StringIO

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import requests

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.h3 import str_to_h3
from megaPLuG.utils.params import build_df_from_dict
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
    # WARNING: This algorithm leaves us with less than 100% VIUS tabulation weight
    # coverage. We're ignoring weight held in vehicles with "Non-registration-state home
    # bases" and "Not Reported" home bases.
    corresp = build_df_from_dict(
        d=params["home_base_corresp"]["values"],
        id_cols=params["home_base_corresp"]["id_columns"],
        value_col="home_base_code",
    )
    scaler = vius.rename(columns={v: k for k, v in params["col_renamer"].items()})
    scaler = scaler.merge(corresp, how="left", on=params["home_source_col"])
    home_in_reg = scaler["home_base_code"] == "Home Base in Register State"
    scaler.loc[home_in_reg, params["id_cols"]["region"]] = scaler.loc[
        home_in_reg, "reg_state"
    ]
    no_home_base = scaler["home_base_code"] == "No Home Base"
    scaler.loc[no_home_base, params["id_cols"]["region"]] = "No Home Base"
    id_cols = list(params["id_cols"].values())
    totals = scaler.groupby(id_cols)[params["totals_col"]].sum()
    totals = totals.reset_index()
    return totals


def format_substation_profiles(profs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Collapse profiles from hour-month combinations to a single characteristic day."""
    pcols = params["columns"]
    profs = profs.rename(columns={v: k for k, v in params["col_renamer"].items()})

    split = profs[pcols["month_hour"]].str.split("_")
    profs[pcols["month"]] = split.transform(lambda x: int(x[0]))
    profs[pcols["hour"]] = split.transform(lambda x: int(x[1]))
    profs = profs.drop(columns=[pcols["month_hour"]])

    # Aggregate to a characteristic day
    profs = profs.groupby([pcols["substation_id"], pcols["hour"]]).agg(
        max_base_by_hour_kw=pd.NamedAgg(pcols["baseload"], "max"),
    )
    profs["max_base_by_hour_mw"] = profs["max_base_by_hour_kw"] / 1000
    profs["max_base_mw"] = profs.groupby(pcols["substation_id"])[
        "max_base_by_hour_mw"
    ].transform(lambda s: s.max())
    profs = profs.drop(columns=["max_base_by_hour_kw"])
    profs = profs.reset_index(pcols["hour"])
    return profs


def format_substation_boundaries(infra: gpd.GeoDataFrame, params: dict) -> pd.DataFrame:
    """Add up substation capacities from the capacities of transformer banks."""
    infra = infra.rename(columns={v: k for k, v in params["col_renamer"].items()})
    infra["substation_id"] = infra["substation_id"].astype(int)

    subs = infra.dissolve(
        by="substation_id",
        aggfunc={
            "substation_name": "first",
            "rating_mw": "sum",
        },
    )
    return subs


def describe_substation_usage(
    profs: pd.DataFrame, subs: gpd.GeoDataFrame, params: dict
) -> pd.DataFrame:
    """Combine baseload profiles and capacities to describe substation usage."""
    pcols = params["columns"]
    subs = subs.drop(columns=params["drop_substation_cols"])
    subs = profs.merge(subs, how="inner", on=pcols["substation_id"])
    subs = subs.reset_index()
    subs = subs.sort_values([pcols["substation_id"], pcols["hour"]])
    subs = subs.set_index(pcols["substation_id"])
    subs[pcols["cap_avail_mw"]] = subs[pcols["rating_mw"]] - subs[pcols["baseload_mw"]]
    return subs
