"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

from megaPLuG.models.charging_algorithms import charge_soc_thresh
from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.params import build_df_from_dict, flatten_dict

logger = logging.getLogger(__name__)

SECS_PER_HOUR = 3600


def set_vehicle_params(vehs: pd.DataFrame, veh_pars: dict, params: dict) -> DwellSet:
    """Set vehicle parameters in advance of simulation."""
    # Seed is based on master seed and vehicle's id to ensure that vehicles are
    # individually controllable without impacting all other vehicles.
    orig_idx = vehs.index.names
    vehs = vehs.reset_index()

    for k, v in veh_pars.items():
        if isinstance(v, dict) and set(v.keys()) == {"id_columns", "values"}:
            # If this is a merge-type param, sensitive to already-defined parameters
            par_df = build_df_from_dict(
                d=v["values"], id_cols=v["id_columns"], value_col=k
            )
            vehs = vehs.merge(par_df, how="left", on=v["id_columns"])
        elif k == "master_seed":
            # Each vehicle gets its own independent seed controlled by the master
            vehs["random_seed"] = vehs["veh_id"] + v
        else:
            # If this is a parameter with the same value across vehicles
            flat = flatten_dict({k: v})
            for col, val in flat.items():
                vehs[col] = val

    vehs = vehs.set_index(orig_idx)
    return vehs


def filter_vehicles(dw: DwellSet, vehs: pd.DataFrame) -> DwellSet:
    """Filter out vehicles that we're not considering."""
    keep_idx = vehs.loc[vehs["has_home_base"], :].index.values
    logger.info("Filter by vehicles by direct dropping")
    old_len = len(dw.data)
    dw.data = dw.data.loc[keep_idx, :]
    new_len = len(dw.data)
    logger.info(f"Rows dropped: {old_len - new_len}, {round(new_len/old_len*100, 1)}%")
    return dw


def filter_dwells(dw: DwellSet, locs: dict, params: dict) -> DwellSet:
    """Set the charging availability for each session."""
    dwell_hrs = dw.data[dw.end] - dw.data[dw.start]
    dw.data["dwell_time_hrs"] = dwell_hrs.dt.total_seconds() / SECS_PER_HOUR
    dw.data["long_dwells"] = dw.data["dwell_time_hrs"] > locs["min_dwell_time_hrs"]

    logger.info("Filter by dwells by accumulating through")
    old_len = len(dw.data)
    dw.data = dw.data.drop(columns=params["drop_cols"])  # Not accumulated
    dw.filter_through("long_dwells")
    new_len = len(dw.data)
    logger.info(f"Rows dropped: {old_len - new_len}, {round(new_len/old_len*100, 1)}%")
    return dw


def mark_locations(dw: DwellSet, veh_locs: pd.DataFrame, params: dict) -> DwellSet:
    """Mark locations-of-interest for each vehicle (e.g. home base)."""
    veh_loc_merge = veh_locs.loc[:, params["veh_loc_cols"]]
    # TODO: Modify this code so that it works with Dask, using a single merge column
    # Achieve this using an encoding function e.g. id1 * set1_size + id2, or a hash
    # Pandas almost surely does this internally, but Dask doesn't presume.
    dw.data = dw.data.merge(veh_loc_merge, how="left", on=[dw.veh, dw.hex])
    return dw


def mark_vehicle_days(dw: DwellSet, params: dict) -> DwellSet:
    """Mark out vehicle-days, the periods between human-intuitive-resets.

    For vehicles with a home base, these would be times between visits to the home base.
    """
    dw.data[params["refresh_col"]] = dw.data[params["loc_col"]].isin(
        params["refresh_locations"]
    )
    dw.data[params["veh_day_col"]] = dw.data.groupby(dw.veh)[
        params["refresh_col"]
    ].transform(lambda ser: ser.shift(1, fill_value=False).cumsum())
    return dw


def mark_critical_days(dw: DwellSet, veh_pars: dict, params: dict) -> DwellSet:
    """Mark critical days, vehicle-days which cannot be achieved on a single charge."""
    crit_days = deepcopy(dw)
    crit_days.filter_through(params["refresh_col"])
    # TODO: Consider switching this to vehicle-specific parameters
    implied_range = (
        veh_pars["battery_capacity_kwh"] / veh_pars["consump_rate_kwh_per_mi"]
    )
    crcol = params["crit_col"]
    crit_days.data[crcol] = crit_days.data[crit_days.dist] > implied_range

    vdcol = params["veh_day_col"]
    crit_days_merge = crit_days.data.loc[:, [vdcol, crcol]]
    dw.data = dw.data.merge(crit_days_merge, how="left", on=[dw.veh, vdcol])
    # Assume a critical day for partial days, since the point of the critical days
    # assumption is to reduce public charging on days when we're sure it's unnecessary
    crit_na = dw.data[crcol].isna()
    dw.data.loc[crit_na, crcol] = True
    dw.data[crcol] = dw.data[crcol].astype(bool)
    return dw


def filter_noncritical_dwells(dw: DwellSet, params: dict) -> DwellSet:
    """Filter out the en-route dwells on non-critical days."""
    dw.data["keep_dwells"] = (
        dw.data[params["refresh_col"]] | dw.data[params["crit_col"]]
    )

    logger.info("Filter by dwells by accumulating through")
    old_len = len(dw.data)
    dw.filter_through("keep_dwells")
    new_len = len(dw.data)
    logger.info(f"Rows dropped: {old_len - new_len}, {round(new_len/old_len*100, 1)}%")
    return dw


def calc_energy_use(dw: DwellSet, params: dict) -> DwellSet:
    """Calculate energy use for all trips."""
    dw.data["energy_use_kwh"] = dw.data[dw.dist] * params["consump_rate_kwh_per_mi"]
    return dw


def set_charging_availability(dw: DwellSet, locs: dict) -> DwellSet:
    """Set the charging availability for each session."""
    dw.data["max_power_kw"] = locs["charging_rate_kw"]
    return dw


def simulate_charging_choice(
    dw: DwellSet, vehs: pd.DataFrame, params: dict
) -> DwellSet:
    """Simulate the charging choices of each vehicle."""
    # TODO: It may be important later to create a function which checks for groupby monotonic increasing.
    if dw.data.index.name != dw.veh:
        raise RuntimeError(
            "The vehicle ID must be the index column for this operation."
        )

    # Allocate columns to fill in, which avoids merging
    dw.data["dwell_start_kwh"] = np.NaN
    dw.data["charge_kwh"] = np.NaN
    tqdm.pandas()
    dw.data = dw.data.groupby(dw.veh, group_keys=False).progress_apply(
        charge_soc_thresh,
        consumed_kwh_col="energy_use_kwh",
        avail_kw_col="max_power_kw",
        dwell_hrs_col="dwell_time_hrs",
        reset_col=dw.reset,
        veh_params=vehs,
        soc_params=params["initial_soc"],
    )
    return dw


def summarize_vehicles(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Summarize the results for each vehicle."""
    dw.data["is_death"] = dw.data[params["dead_energy_col"]] < 0
    n_deaths = dw.data.groupby(dw.veh)["is_death"].sum()
    n_deaths.name = "n_deaths"
    vehs = vehs.merge(n_deaths, how="inner", on=dw.veh)

    logger.info("Deaths per vehicle:")
    logger.info(n_deaths.describe())
    return vehs


def get_hex_events_from_dwells(dw: DwellSet, params: dict) -> pd.DataFrame:
    """Convert vehicle dwells to hexagon events."""
    hex_kw_cols = [f"{seqn}_hex_kw_diff" for seqn in params["seq_names"]]
    dw.data[hex_kw_cols[0]] = dw.data["charge_kwh"] / dw.data["dwell_time_hrs"]
    dw.data[hex_kw_cols[1]] = -dw.data[hex_kw_cols[0]]

    dw.data = dw.data.dropna(subset=hex_kw_cols)
    dw.seq_names = params["seq_names"]
    events = dw.to_hex_profiles()
    return events
