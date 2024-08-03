"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

import logging

import numpy as np
from tqdm import tqdm

from megaPLuG.models.charging_algorithms import charge_soc_thresh
from megaPLuG.models.dwell_sets import DwellSet

logger = logging.getLogger(__name__)


def calc_energy_use(dw: DwellSet, params: dict) -> DwellSet:
    """Calculate energy use for all trips."""
    dw.data["energy_use_kwh"] = dw.data[dw.dist] * params["consump_rate_kwh_per_mi"]
    return dw


def set_charging_availability(dw: DwellSet, vehs: dict, locs: dict) -> DwellSet:
    """Set the charging availability for each session."""
    dw.data["dwell_time_hrs"] = (
        dw.data[dw.end] - dw.data[dw.start]
    ).dt.total_seconds() / 3600
    dw.data["long_dwells"] = dw.data["dwell_time_hrs"] > locs["min_dwell_time_hrs"]

    logger.info("Filter by accumulating through")
    old_len = dw.data.shape[0]
    dw.filter_through("long_dwells")
    new_len = dw.data.shape[0]
    logger.info(f"Rows dropped: {old_len - new_len}, {round(new_len/old_len*100, 1)}%")
    dw.data["max_power_kw"] = vehs["charging_rate_kw"]
    return dw


def simulate_charging_choice(dw: DwellSet, params: dict) -> DwellSet:
    """Simulate the charging choices of each vehicle."""
    # TODO: It may be important later to create a function which checks for groupby monotonic increasing.
    logger.info("Set independent, controlled random seeds for each vehicle")
    if dw.data.index.name != dw.veh:
        raise RuntimeError(
            "The vehicle ID must be the index column for this operation."
        )
    veh_ids = dw.data.index.unique()
    # Seed is based on master seed and vehicle's id to ensure that vehicles are
    # individually controllable without impacting all other vehicles.
    veh_rngs = {id: np.random.default_rng(seed=params["seed"] + id) for id in veh_ids}

    logger.info("Conduct charging simulation through groupby-apply")
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
        batt_cap_kwh=params["battery_capacity_kwh"],
        soc_pars=params["initial_soc"],
        charge_soc=params["charge_soc_thresh"],
        rngs=veh_rngs,
    )
    n_vehs = veh_ids.shape[0]
    n_dead = dw.data.groupby(dw.veh)["dwell_start_kwh"].last(skipna=False).isna().sum()
    n_elect = n_vehs - n_dead
    pct_elect = round(n_elect / n_vehs * 100, 1)
    logger.info(f"Electrifiable vehicles: {n_elect}, {pct_elect}%")
    return dw
