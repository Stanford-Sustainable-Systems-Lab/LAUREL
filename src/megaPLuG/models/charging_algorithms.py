import numpy as np
import pandas as pd
from numba import njit


def charge_soc_thresh(
    grp: pd.DataFrame,
    consumed_kwh_col: str,
    avail_kw_col: str,
    dwell_hrs_col: str,
    reset_col: str,
    batt_cap_kwh: float,
    soc_pars: dict,
    charge_soc: float,
    rngs: dict[int, np.random.Generator],
) -> pd.DataFrame:
    """Execute the charging strategy of charging below an SoC threshold."""
    arr = _charge_soc_thresh_core(
        consumed_kwh=grp[consumed_kwh_col].values,
        avail_kw=grp[avail_kw_col].values,
        dwell_hrs=grp[dwell_hrs_col].values,
        reset=grp[reset_col].values,
        batt_cap=batt_cap_kwh,
        rng=rngs[grp.name],  # This assumes the result of a groupby
        rng_params=np.array([soc_pars["alpha"], soc_pars["beta"]]),
        charge_soc=charge_soc,
    )
    grp.loc[:, "dwell_start_kwh"] = arr[:, 0]
    grp.loc[:, "charge_kwh"] = arr[:, 1]
    return grp


@njit
def _charge_soc_thresh_core(
    consumed_kwh: np.ndarray[float],
    avail_kw: np.ndarray[float],
    dwell_hrs: np.ndarray[float],
    reset: np.ndarray[bool],
    batt_cap: float,
    rng: np.random.Generator,
    rng_params: np.ndarray[float],
    charge_soc: float,
) -> np.ndarray:
    """Execute the charging strategy of charging below an SoC threshold."""
    if not consumed_kwh.shape == avail_kw.shape == dwell_hrs.shape:
        raise RuntimeError("The three arrays must have the same shape.")
    if not reset[0]:
        raise RuntimeError("The first observation must have an SoC reset.")
    energy_tracker = np.empty((consumed_kwh.shape[0], 2))
    charge_kwh = 1
    dwell_init_kwh = 0
    dead = False
    for i in range(energy_tracker.shape[0]):
        if reset[i]:
            soc = rng.beta(a=rng_params[0], b=rng_params[1])
            cur_energy = batt_cap * soc
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
