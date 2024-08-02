"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

import logging

import numpy as np
import pandas as pd
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
    dw.filter_through("long_dwells")
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
    # Allocate columns to fill in, which avoids merging
    dw.data["dwell_init_kwh"] = np.NaN
    dw.data["charge_kwh"] = np.NaN
    tqdm.pandas()
    dw.data = dw.data.groupby(dw.veh, group_keys=False).progress_apply(
        charge_soc_thresh,
        consumed_kwh_col="energy_use_kwh",
        avail_kw_col="max_power_kw",
        dwell_hrs_col="dwell_time_hrs",
        batt_cap_kwh=params["battery_capacity_kwh"],
        soc_inits=soc_inits,
        charge_soc=params["charge_soc_thresh"],
    )
    return dw
