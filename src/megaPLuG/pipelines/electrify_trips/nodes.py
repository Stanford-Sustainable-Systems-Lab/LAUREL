"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

import logging

import numpy as np
import pandas as pd
from numba import njit

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
    dw.filter_through(
        "long_dwells"
    )  # This is introducing NaNs in the first element of trip_miles of some groups, and is shockingly slow. Let's fix both of those.
    dw.data["max_power_kw"] = vehs["charging_rate_kw"]
    return dw


def simulate_charging_choice(dw: DwellSet, params: dict) -> DwellSet:
    """Simulate the charging choices of each vehicle."""
    # TODO: It may be important later to create a function which checks for groupby monotonic increasing.
    logger.info("Sample initial states of charge")
    soc_pars = params["initial_soc"]
    rng = np.random.default_rng(seed=soc_pars["seed"])
    if dw.data.index.name != dw.veh:
        raise RuntimeError(
            "The vehicle ID must be the index column for this operation."
        )
    veh_ids = dw.data.index.unique()
    soc_inits = rng.beta(a=soc_pars["alpha"], b=soc_pars["beta"], size=veh_ids.shape[0])
    soc_inits = pd.Series(data=soc_inits, index=veh_ids)

    logger.info("Conduct charging simulation through groupby-apply")
    charges = dw.data.groupby(dw.veh, group_keys=False).apply(
        lambda grp: pd.DataFrame(
            charge_soc_fast(
                consumed_kwh=grp["energy_use_kwh"].values,
                avail_kw=grp["max_power_kw"].values,
                dwell_hrs=grp["dwell_time_hrs"].values,
                batt_cap=params["battery_capacity_kwh"],
                soc_init=soc_inits[grp.name],
                charge_soc=params["charge_soc_thresh"],
            ),
            index=grp.index,
            columns=["dwell_init_kwh", "charge_kwh"],
        )
    )

    logger.info("Merge charging simulation results back onto dwells")
    if np.all(dw.data.index == charges.index):
        dw.data = pd.concat([dw.data, charges], axis=1)
    else:
        dw.data = dw.data.merge(charges, how="left", left_index=True, right_index=True)
    return dw


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
