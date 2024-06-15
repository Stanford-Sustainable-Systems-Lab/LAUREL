"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import logging
import re
from itertools import product

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def calc_derived_trip_cols(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Calculate derived variables which are needed for events."""
    trips = trips.reset_index()
    trips = trips.rename(columns=params["initial_rename"])

    dt = pd.to_timedelta(trips["dwell_time_hrs"] * 3600, unit="s")
    trips["dwell_end_timestamp_utc"] = trips["dwell_start_timestamp_utc"] + dt
    trips["dwell_end_veh_kwh"] = trips["dwell_start_veh_kwh"] + trips["charge_kwh"]
    trips["dwell_start_hex_kw_diff"] = trips["charge_kwh"] / trips["dwell_time_hrs"]
    trips["dwell_end_hex_kw_diff"] = -trips["dwell_start_hex_kw_diff"]
    return trips


def get_events_from_trips(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Unpack trips (drive + dwell) into individual time-space transition events."""
    # Set new column MultiIndex to prepare for stacking
    trips.set_index(params["id_cols"], inplace=True)
    keep_cols = []
    tups = []
    for i, s in enumerate(params["seq_names"]):
        orig = [col for col in trips.columns if s in col]
        tails = [re.findall(f"(?<={s}_).+", c)[0] for c in orig]
        tups.extend(product(tails, [s], [i]))
        keep_cols.extend(orig)
    idx = pd.MultiIndex.from_tuples(tups, names=["variable", "seq_name", "seq_id"])
    trips = trips[keep_cols]
    trips.columns = idx

    # Stack the trips into events
    events = trips.stack(level=["seq_id", "seq_name"], future_stack=True)
    events.index = events.index.droplevel("seq_id")
    events["event_id"] = pd.RangeIndex(0, len(events))
    events = events.set_index("event_id", append=True)
    idx_names = ["event_id"] + params["id_cols"] + ["seq_name"]
    events.index = events.index.reorder_levels(idx_names)
    return events


def aggregate_regional_loads(
    sessions: dd.DataFrame,
    grid_regions: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Aggregate loads within grid impact regions."""
    raise NotImplementedError()


def plot_peak_load_evolution(
    vehicle_load: pd.DataFrame,
    baseline_load: pd.DataFrame,
) -> Figure:
    """Plot baseline loads compared to baseline plus vehicles loads."""
    raise NotImplementedError()


def plot_hourly_load(
    vehicle_load: pd.DataFrame,
    baseline_load: pd.DataFrame,
) -> Figure:
    """Plot baseline loads compared to baseline plus vehicles loads."""
    raise NotImplementedError()
