"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from numba import njit

logger = logging.getLogger(__name__)


def convert_to_pandas(trips: dd.DataFrame) -> pd.DataFrame:
    trips = trips.compute()
    return trips


def calc_energy_use(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Calculate energy use for all trips."""
    trips["energy_use_kwh"] = (
        trips[params["dist_col"]] * params["consump_rate_kwh_per_mi"]
    )
    return trips


def calc_dwell_hrs(trips: dd.DataFrame) -> dd.DataFrame:
    """Calculate the length of time at each dwell."""
    # with ProgressBar():
    #     test = trips.compute()
    trips["dwell_time_hrs"] = (
        trips.groupby("vehicle_id")["start_timestamp_utc"].shift(-1)
        - trips["end_timestamp_utc"]
    )
    trips["dwell_time_hrs"] = trips["dwell_time_hrs"].dt.total_seconds() / 3600

    # meta = pd.Series(
    #     data=pd.Timedelta(1),
    #     index=pd.RangeIndex(start=0, stop=1),
    #     name="dwell_time_hrs",
    # )

    # trips["dwell_time_hrs"] = trips.groupby(trips.index).apply(
    #     lambda grp: grp["start_timestamp_utc"].shift(-1) - grp["end_timestamp_utc"],
    #     meta=meta,
    # )
    # trips["dwell_time_hrs"] = trips["dwell_time_hrs"].dt.total_seconds() / 3600
    # with ProgressBar():
    #     test=trips.compute()
    return trips


def simulate_charging_choice(trips: pd.DataFrame, params: dict) -> dd.DataFrame:
    """Simulate the charging choices of each vehicle."""
    logger.info("Calculate new columns")
    trips["max_power_kw"] = 350

    logger.info("Sample initial states of charge")
    soc_pars = params["initial_soc"]
    rng = np.random.default_rng(seed=soc_pars["seed"])
    veh_ids = trips["vehicle_id"].unique()  # .compute()
    soc_inits = rng.beta(a=soc_pars["alpha"], b=soc_pars["beta"], size=veh_ids.shape[0])
    soc_inits = pd.Series(data=soc_inits, index=veh_ids)

    # Switch trips index to vehicle_id (already sorted) to speed up groupby
    logger.info("Conduct charging simulation through groupby-apply")
    trips = trips.reset_index(drop=False)
    trips = trips.set_index("vehicle_id", drop=True)  # , sorted=True)
    charges = trips.groupby(trips.index, group_keys=False).apply(
        lambda grp: pd.DataFrame(
            charge_soc_fast(
                consumed_kwh=grp["energy_use_kwh"].values,
                avail_kw=grp["max_power_kw"].values,
                dwell_hrs=grp["dwell_time_hrs"].values,
                batt_cap=params["battery_capacity_kwh"],
                soc_init=soc_inits[grp.name],
                charge_soc=params["charge_soc"],
            ),
            index=grp["veh_time_id"],
            columns=["charge_kwh", "dwell_init_kwh"],
        ),
        # meta={"charge_kwh": 'f8', "dwell_init_kwh": 'f8'},
    )
    with ProgressBar():
        charges = charges.compute()  # Note: Merging breaks if this is not pre-computed to a pandas DataFrame, but this might cause memory issues later

    # Switch trips index back to veh_time_id (already sorted) to speed up merge
    logger.info("Merge charging simulation results back onto trips")
    trips = trips.reset_index(drop=True)
    trips = trips.set_index("veh_time_id", drop=True)  # , sorted=True)
    trips_w_charge = trips.merge(charges, how="left", left_index=True, right_index=True)
    with ProgressBar():
        trips_w_charge = trips_w_charge.compute()
    if isinstance(trips_w_charge, pd.DataFrame):
        trips_w_charge = dd.from_pandas(trips_w_charge, npartitions=2)
    return trips_w_charge


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
    charge_kwh = 0
    dwell_init_kwh = 1
    cur_energy = batt_cap * soc_init
    for i in np.arange(energy_tracker.shape[0]):
        cur_energy -= consumed_kwh[i]
        energy_tracker[i, dwell_init_kwh] = cur_energy
        if cur_energy / batt_cap <= charge_soc:
            chg = np.minimum(batt_cap - cur_energy, dwell_hrs[i] * avail_kw[i])
        else:
            chg = 0
        energy_tracker[i, charge_kwh] = chg
        cur_energy += chg
    return energy_tracker
