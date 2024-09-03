import numpy as np
import pandas as pd
from numba import njit


def charge_soc_thresh(
    grp: pd.DataFrame,
    consumed_kwh_col: str,
    avail_kw_col: str,
    dwell_hrs_col: str,
    reset_col: str,
    veh_params: pd.DataFrame,
    out_cols: list[str],
) -> pd.DataFrame:
    """Execute the charging strategy of charging below an SoC threshold."""
    cur_veh = veh_params.loc[grp.name]
    cur_rng = np.random.default_rng(seed=cur_veh["random_seed"])
    rng_params = np.array([cur_veh["initial_soc_alpha"], cur_veh["initial_soc_beta"]])
    arr = _charge_soc_thresh_core(
        consumed_kwh=grp[consumed_kwh_col].values,
        avail_kw=grp[avail_kw_col].values,
        dwell_hrs=grp[dwell_hrs_col].values,
        reset=grp[reset_col].values,
        batt_cap=cur_veh["battery_capacity_kwh"],
        charge_soc=cur_veh["charge_soc_thresh"],
        rng=cur_rng,
        rng_params=rng_params,
    )
    for i, col in enumerate(out_cols):
        grp.loc[:, col] = arr[:, i]
    return grp


@njit
def _charge_soc_thresh_core(
    consumed_kwh: np.ndarray[float],
    avail_kw: np.ndarray[float],
    dwell_hrs: np.ndarray[float],
    reset: np.ndarray[bool],
    batt_cap: float,
    charge_soc: float,
    rng: np.random.Generator,
    rng_params: np.ndarray[float],
) -> np.ndarray:
    """Execute the charging strategy of charging below an SoC threshold."""
    nsteps = consumed_kwh.shape[0]
    cur_energy = np.NaN
    energy_tracker = np.empty((nsteps, 2))
    dwell_init_kwh, charge_kwh = 0, 1  # Set energy tracker index names
    for i in range(nsteps):
        if reset[i]:
            soc = rng.beta(a=rng_params[0], b=rng_params[1])
            cur_energy = batt_cap * soc
        cur_energy -= consumed_kwh[i]
        energy_tracker[i, dwell_init_kwh] = cur_energy
        avail_kwh = dwell_hrs[i] * avail_kw[i]
        if np.isnan(cur_energy) or cur_energy < 0:  # Currently dead
            if avail_kwh >= batt_cap:  # If full recharge is possible, then refresh
                cur_energy = 0
                chg = batt_cap
            else:  # If not, then become/stay dead
                chg = np.NaN
        elif cur_energy / batt_cap <= charge_soc:
            chg = np.minimum(batt_cap - cur_energy, avail_kwh)
        else:
            chg = 0
        energy_tracker[i, charge_kwh] = chg
        cur_energy += chg

    return energy_tracker
