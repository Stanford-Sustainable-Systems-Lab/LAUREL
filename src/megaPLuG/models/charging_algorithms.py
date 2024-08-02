import numpy as np
import pandas as pd
from numba import njit


def charge_soc_thresh(
    grp: pd.DataFrame,
    consumed_kwh_col: str,
    avail_kw_col: str,
    dwell_hrs_col: str,
    batt_cap_kwh: float,
    soc_inits: pd.DataFrame,
    charge_soc: float,
) -> pd.DataFrame:
    """Execute the charging strategy of charging below an SoC threshold."""
    arr = _charge_soc_thresh_core(
        consumed_kwh=grp[consumed_kwh_col].values,
        avail_kw=grp[avail_kw_col].values,
        dwell_hrs=grp[dwell_hrs_col].values,
        batt_cap=batt_cap_kwh,
        soc_init=soc_inits[grp.name],  # This assumes the result of a groupby
        charge_soc=charge_soc,
    )
    grp.loc[:, "dwell_start_kwh"] = arr[:, 0]
    grp.loc[:, "charge_kwh"] = arr[:, 1]
    return grp


@njit
def _charge_soc_thresh_core(
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
