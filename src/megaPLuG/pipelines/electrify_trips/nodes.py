"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

import logging

import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)


def calc_energy_use(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Calculate energy use for all trips."""
    trips["energy_use_kwh"] = (
        trips[params["dist_col"]] * params["consump_rate_kwh_per_mi"]
    )
    return trips


def calc_dwell_hrs(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Calculate the length of time at each dwell."""
    if trips.index.name != params["index_name"]:
        raise RuntimeError("Index name does not match the name from the config.")
    if not trips.index.is_monotonic_increasing:
        raise RuntimeError("Index must already be monotonic increasing.")

    # TODO: To use this within Dask (if datasets get bigger) then simply run the rest of
    # this function within a dask.dataframe.map_partitions call. This would require
    # earlier partitioning based on vehicle.
    trips["dwell_time_hrs"] = (
        trips.groupby(params["group_col_name"])["start_timestamp_utc"].shift(-1)
        - trips["end_timestamp_utc"]
    )
    trips["dwell_time_hrs"] = trips["dwell_time_hrs"].dt.total_seconds() / 3600
    drop_idx = trips[trips["dwell_time_hrs"].isna()].index
    trips = trips.drop(index=drop_idx)  # To get rid of NaT value from shift
    return trips


def set_charging_availability(trips: pd.DataFrame) -> pd.DataFrame:
    """Set the charging availability for each session."""
    trips["max_power_kw"] = 350
    return trips


def simulate_charging_choice(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Simulate the charging choices of each vehicle."""
    if trips.index.name != params["index_name"]:
        raise RuntimeError("Index name does not match the name from the config.")
    if not trips.index.is_monotonic_increasing:
        raise RuntimeError("Index must already be monotonic increasing.")

    logger.info("Sample initial states of charge")
    soc_pars = params["initial_soc"]
    rng = np.random.default_rng(seed=soc_pars["seed"])
    veh_ids = trips["vehicle_id"].unique()
    soc_inits = rng.beta(a=soc_pars["alpha"], b=soc_pars["beta"], size=veh_ids.shape[0])
    soc_inits = pd.Series(data=soc_inits, index=veh_ids)

    # Switch trips index to vehicle_id (already sorted) to speed up groupby
    logger.info("Conduct charging simulation through groupby-apply")
    charges = trips.groupby(params["veh_id_col"], group_keys=False).apply(
        lambda grp: pd.DataFrame(
            charge_soc_fast(
                consumed_kwh=grp["energy_use_kwh"].values,
                avail_kw=grp["max_power_kw"].values,
                dwell_hrs=grp["dwell_time_hrs"].values,
                batt_cap=params["battery_capacity_kwh"],
                soc_init=soc_inits[grp.name],
                charge_soc=params["charge_soc"],
            ),
            index=grp.index,
            columns=["dwell_init_kwh", "charge_kwh"],
        )
    )

    logger.info("Merge charging simulation results back onto trips")
    if np.all(trips.index == charges.index):
        trips = pd.concat([trips, charges], axis=1)
    else:
        trips = trips.merge(charges, how="left", left_index=True, right_index=True)
    return trips


@njit
def charge_soc_fast(
    consumed_kwh: np.ndarray,
    avail_kw: np.ndarray,
    dwell_hrs: np.ndarray,
    batt_cap: float,
    soc_init: float,
    charge_soc: float,
) -> np.ndarray:
    """Execute the charging strategy of charging below an SoC threshold."""
    if not consumed_kwh.shape == avail_kw.shape == dwell_hrs.shape:
        raise RuntimeError("The three arrays must have the same shape.")
    energy_tracker = np.empty((consumed_kwh.shape[0], 2))
    charge_kwh = 1
    dwell_init_kwh = 0
    dead = False
    cur_energy = batt_cap * soc_init
    for i in np.arange(energy_tracker.shape[0]):
        cur_energy -= consumed_kwh[i]
        energy_tracker[i, dwell_init_kwh] = cur_energy
        if cur_energy < 0:
            dead = True
            break
        if cur_energy / batt_cap <= charge_soc:
            chg = np.minimum(batt_cap - cur_energy, dwell_hrs[i] * avail_kw[i])
        else:
            chg = 0
        energy_tracker[i, charge_kwh] = chg
        cur_energy += chg

    if dead:
        energy_tracker[(i + 1) :, dwell_init_kwh] = np.NaN
        energy_tracker[i:, charge_kwh] = np.NaN
    return energy_tracker
