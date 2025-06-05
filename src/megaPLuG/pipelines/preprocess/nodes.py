"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

import logging
import re
from io import StringIO

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import h3.api.numpy_int as h3
import numpy as np
import pandas as pd
import requests
from dask.diagnostics import ProgressBar
from sklearn.preprocessing import OneHotEncoder

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.geo import METERS_PER_MILE
from megaPLuG.utils.h3 import H3_DEFAULT_RESOLUTION, str_to_h3
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


def calc_derived_trip_cols(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Calculate derived variables which are needed for events."""
    trips["trip_hrs"] = total_hours(
        trips[params["time_cols"]["trip_end"]]
        - trips[params["time_cols"]["trip_start"]]
    )

    if params["persist"]:
        trips = trips.persist()
    return trips


def prepare_stop_locations(parks: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Prepare the stop locations for optional stops."""
    pcols = params["columns"]
    parks[pcols["hex"]] = parks.geometry.apply(
        lambda pt: h3.latlng_to_cell(pt.y, pt.x, res=H3_DEFAULT_RESOLUTION)
    )
    parks = parks.rename_geometry(pcols["park_point"])
    parks[pcols["park_id"]] = pd.RangeIndex(stop=parks.shape[0])
    parks = parks.loc[:, params["keep_cols"]]
    return parks


def get_optional_stop_trips(
    routes: dgpd.GeoDataFrame, parks: gpd.GeoDataFrame, params: dict
) -> pd.DataFrame:
    """Compute the optional trip stops and their distances along the routes."""
    pcols = params["columns"]

    # Set up the parks for spatial join
    parks = parks.to_crs(params["projected_crs"])
    parks["buffer"] = parks.geometry.buffer(
        distance=params["park_buffer_miles"] * METERS_PER_MILE
    )
    parks = parks.set_geometry("buffer")

    # Eliminate routes with no geometry and unused columns
    trips_source = routes.dropna(subset=[pcols["route_geom"]])
    trips_source = trips_source.drop(columns=params["drop_cols_initial"])

    # Spatial join
    trips_short = trips_source.to_crs(params["projected_crs"])
    trips_short = trips_short.sjoin(parks, how="inner", predicate="intersects")
    trips_short = trips_short.drop(columns=["index_right"])

    # Find distances along the route for each optional stop
    def project_partition(
        part: gpd.GeoDataFrame, line_col: str, point_col: str, out_col: str
    ) -> gpd.GeoDataFrame:
        """Project the points_col of the partition on to the line_col."""
        part[out_col] = part[line_col].project(part[point_col]) / METERS_PER_MILE
        return part

    trips_short[pcols["dist_along_miles"]] = np.nan
    trips_short = trips_short.map_partitions(
        project_partition,
        line_col=pcols["route_geom"],
        point_col=pcols["park_point"],
        out_col=pcols["dist_along_miles"],
        meta=trips_short,
    )

    # After this point, the route geometries are no longer needed, so we drop them to save memory
    trips_short[pcols["hex_end"]] = trips_short[pcols["hex_park"]]
    trips_short = trips_short.drop(
        columns=[pcols["route_geom"], pcols["park_point"], pcols["hex_park"]]
    )

    # Prepare the original trips for concatenation
    trips_source["dist_along_miles"] = trips_source["trip_miles_route"]
    trips_orig = trips_source.drop(columns=pcols["route_geom"])

    logger.info("Computing the optional stop trips by spatial joining and projecting.")
    with ProgressBar():
        trips_short, trips_orig = dd.compute(trips_short, trips_orig)

    # Concatenate and format original and new short trips
    concatter = {False: trips_orig, True: trips_short}
    trips_mod = pd.concat(concatter, axis=0, names=[pcols["is_optional"]])
    trips_mod = trips_mod.reset_index(pcols["is_optional"])
    return trips_mod


def describe_optional_stop_trips(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Describe the new timings and distances for optional and original trips."""
    pcols = params["columns"]

    # Eliminate optional trips too close to the ends of the original trip
    started_at_park = trips[pcols["dist_along_miles"]] < params["park_buffer_miles"]
    ended_at_park = trips[pcols["dist_along_miles"]] > (
        trips[pcols["miles_route"]] - params["park_buffer_miles"]
    )
    is_opt = trips[pcols["is_optional"]]
    trips = trips.loc[(~started_at_park & ~ended_at_park & is_opt) | ~is_opt, :]

    logger.info("Sort trips to enable position-based computations")
    trip_id_cols = params["trip_id_cols"]
    trips = trips.sort_values(
        trip_id_cols + [pcols["dist_along_miles"]], ascending=True
    )

    logger.info("Compute new timings and distances for optional and original trips")
    # Distances by segment
    trips["dist_prev_miles"] = trips.groupby(trip_id_cols)[
        pcols["dist_along_miles"]
    ].shift(1, fill_value=0.0)
    trips["trip_miles_route_seg"] = (
        trips[pcols["dist_along_miles"]] - trips["dist_prev_miles"]
    )

    # Times by segment
    trips["trip_hrs_route_seg"] = (
        trips["trip_miles_route_seg"] / trips[pcols["speed_route"]]
    )
    # TODO: Uncomment these lines once we pass `trip_hrs` through the routing
    # time_scaler = trips[pcols["hours_orig"]] / trips[pcols["hours_route"]]
    # trips["trip_hrs_route_seg"] = trips["trip_hrs_route_seg"] * time_scaler
    trips["trip_time_route_seg"] = pd.to_timedelta(
        trips["trip_hrs_route_seg"], unit="h"
    )
    trips["time_shift"] = trips.groupby(trip_id_cols)["trip_time_route_seg"].cumsum()

    trips["new_end"] = trips[pcols["start_time"]] + trips["time_shift"]
    trips["new_end"] = trips["new_end"].dt.round("s")
    trips["new_start"] = trips.groupby(trip_id_cols)["new_end"].shift(1)
    trips["new_start"] = trips["new_start"].fillna(trips[pcols["start_time"]])

    # Format to match original trips dataset
    drop_col_set = set(params["rename_cols_final"].keys())
    drop_col_set = drop_col_set.intersection(trips.columns)
    trips = trips.drop(columns=drop_col_set)
    trips_out = trips.rename(
        columns={v: k for k, v in params["rename_cols_final"].items()}
    )
    trips_out = trips_out.loc[:, params["keep_cols_final"]]
    return trips_out


def concat_optional_stops(
    trips_orig: dd.DataFrame, trips_opt: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Concatenate new optional trips onto original trips."""
    logger.info("Computing original trips into memory.")
    trips_orig = trips_orig.drop(columns=params["drop_cols"])
    trips_orig = trips_orig.compute()

    logger.info("Concatenating and sorting trips.")
    concatter = {True: trips_orig, False: trips_opt}
    trips = pd.concat(concatter, axis=0, names=["is_original"])
    trips = trips.reset_index("is_original")

    # Drop the original versions of trips which have been split
    sort_cols = params["trip_id_cols"] + [params["dist_col"]]
    trips = trips.sort_values(sort_cols, ascending=True)
    # We keep the first trip because the split, optional trips are guaranteed to be shorter than the original trips
    trips = trips.drop_duplicates(subset=params["trip_id_cols"], keep="first")
    trips = trips.drop(columns=["is_original"])
    return trips


def strip_vehicle_attrs(
    trips: dd.DataFrame, params: dict
) -> tuple[dd.DataFrame, pd.DataFrame]:
    """Get vehicle-specific attributes which stay constant."""
    n_trips_by_veh = trips[params["veh_id_col"]].value_counts().compute()
    drop_idx = n_trips_by_veh.loc[n_trips_by_veh < params["min_trips_per_veh"]].index

    veh_cols = [params["veh_id_col"]] + params["veh_attr_cols"]
    vehs = trips.loc[:, veh_cols].drop_duplicates().compute()
    vehs = vehs.set_index(params["veh_id_col"]).sort_index()
    vehs = vehs.drop(index=drop_idx)
    return vehs


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
