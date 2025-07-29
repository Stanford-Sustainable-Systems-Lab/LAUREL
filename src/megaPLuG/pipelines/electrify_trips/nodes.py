"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

import logging

import numpy as np
import pandas as pd

from megaPLuG.models.charging_algorithms import ForwardLookingChargingChoiceStrategy
from megaPLuG.models.dwell_sets import CumAggFunc, DwellSet
from megaPLuG.models.manage_charging import _MANAGER_MAP
from megaPLuG.utils.data import merge_dataframes_node
from megaPLuG.utils.params import build_df_from_dict
from megaPLuG.utils.time import total_hours

logger = logging.getLogger(__name__)


def filter_vehicles(dw: DwellSet, vehs: pd.DataFrame) -> DwellSet:
    """Filter out vehicles that we're not considering."""
    logger.info("Filter by vehicles by direct dropping")
    old_len = len(dw.data)

    keep_idx = np.intersect1d(dw.data.index.values, vehs.index.values)
    num_no_dwell_vehs = np.setdiff1d(vehs.index.values, keep_idx).size
    if num_no_dwell_vehs > 0:
        logger.warning(
            f"{num_no_dwell_vehs} vehicles were not found in the dwell data."
        )
    dw.data = dw.data.loc[keep_idx]

    new_len = len(dw.data)
    abs_diff = old_len - new_len
    pct_diff = round(abs_diff / old_len * 100, 1)
    logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")
    return dw


def calc_dwell_durations(dw: DwellSet, params: dict) -> DwellSet:
    """Mark dwells which could provide substantial SoC to each vehicle."""
    iocols = params["in_out_time_cols"]
    for col in iocols.values():
        dw.data[col] = pd.to_timedelta(dw.data[col], unit=params["in_out_time_unit"])

    # Adjust dwell start and end times to allow time for vehicle to plug in and out
    # If the stop is optional (proxied by identical start and end times), then leave
    # its duration as zero.
    is_optional = dw.data[dw.start] == dw.data[dw.end]
    dw.data[dw.end] -= ~is_optional * dw.data[iocols["plug_out"]]
    dw.data[dw.start] += ~is_optional * dw.data[iocols["plug_in"]]

    dwell_time_col = params["dwell_time_col"]
    dw.data[dwell_time_col] = total_hours(dw.data[dw.end] - dw.data[dw.start])
    return dw


def prepare_modes(modes: dict) -> pd.DataFrame:
    """Prepare the modes dataframe."""
    name_col = modes.pop("name_column")
    id_col = modes.pop("id_column")
    modes = pd.DataFrame.from_dict(data=modes, orient="index")
    modes.index.name = name_col
    modes = modes.reset_index()
    modes.index.name = id_col
    return modes


def prepare_mode_loc_corresp(modes: pd.DataFrame, params: dict) -> DwellSet:
    """Assign modes to each dwell using a boolean vector of mode availability."""
    avail_dict = params["charge_modes_avail"]
    avails = build_df_from_dict(
        d=avail_dict["values"],
        id_cols=avail_dict["id_columns"],
        value_col=avail_dict["value_col"],
    )

    poss = modes["mode_name"]
    avails[params["value_col_bool"]] = avails[avail_dict["value_col"]].transform(
        lambda av: np.isin(poss, av)
    )
    avails[params["loc_col"]] = pd.Categorical(avails[params["loc_col"]])
    avails[params["max_power_col"]] = avails[params["value_col_bool"]].transform(
        lambda a: np.max(a * modes[params["max_power_source_col"]])
    )
    return avails


def merge_dwellset_node(dw: DwellSet, right: pd.DataFrame, params: dict) -> DwellSet:
    """Use merge_dataframes_node on a DwellSet."""
    dw.data = merge_dataframes_node(
        left=dw.data,
        right=right,
        params=params,
    )
    return dw


def calc_energy_use(dw: DwellSet, params: dict) -> DwellSet:
    """Calculate energy use for all trips."""
    dw.data[params["energy_col"]] = (
        dw.data[dw.trip_dist] * dw.data[params["consump_col"]]
    )
    return dw


def mark_shift_refreshes(dw: DwellSet, params: dict) -> DwellSet:
    """Mark shifts by setting a 'refresh' column.

    This is done using a time threshold, currently based on the Federal Motor Carrier
    Safety Administration (FMCSA) hours of service regulations for commercial vehicle
    drivers.
    """
    pcols = params["columns"]
    dw.data[pcols["refresh"]] = dw.data[pcols["dur"]] >= params["min_refresh_hrs"]
    return dw


def mark_critical_days(dw: DwellSet, params: dict) -> DwellSet:
    """Mark critical days, vehicle-days which cannot be achieved on a single charge."""
    refr_col = params["refresh_col"]
    crit_bnd_col = params["crit_bound_col"]

    hrs_to_fill = dw.data[params["batt_cap_col"]] / dw.data[params["max_power_col"]]
    can_fully_charge = dw.data[params["dur_col"]] >= hrs_to_fill
    dw.data[crit_bnd_col] = dw.data[refr_col] & can_fully_charge

    nrg_col_next = params["energy_col_next_trip"]
    nrg_col_shift = params["energy_col_remain_shift"]
    fill_val = dw.data[params["energy_col"]].mean()  # Imputing shifted values
    dw.data[nrg_col_next] = dw.data.groupby(dw.veh)[params["energy_col"]].shift(
        -1, fill_value=fill_val
    )
    dw.accum_masked(
        crit_bnd_col,
        accum_cols=nrg_col_next,
        reverse=True,
        write_all=True,
        inplace=True,
    )
    dw.data = dw.data.rename(columns={f"{nrg_col_next}_{crit_bnd_col}": nrg_col_shift})

    # Apply vehicle-specific battery capacity
    crcol = params["crit_col"]
    dw.data[crcol] = dw.data[nrg_col_shift] > dw.data[params["batt_cap_col"]]

    # Assume a critical day for partial days, since the point of the critical days
    # assumption is to reduce public charging on days when we're sure it's unnecessary
    # Note that "boolean" is different from "bool" data type. See Pandas BooleanArray
    is_critical = dw.data[crcol].astype("boolean")
    is_crit_bnd = dw.data[crit_bnd_col].astype("boolean")
    dw.data[crcol] = (is_crit_bnd & is_critical) ^ ~(is_crit_bnd | pd.NA)
    dw.data[crcol] = dw.data[crcol].groupby(dw.veh, sort=False).ffill()
    dw.data[crcol] = dw.data[crcol].fillna(True).astype(bool)
    # dw.data[refr_col] = dw.data[refr_col].fillna(False).astype(bool)
    dw.data.drop(columns=[crit_bnd_col], inplace=True)
    return dw


def filter_dwells(dw: DwellSet, params: dict) -> DwellSet:
    """Filter out the en-route dwells on non-critical days.

    We drop dwells wich are:
        - not a refresh location OR not enroute during a critical day
        - AND which have a duration longer than the plug_in + plug_out time
    """
    logger.info("Filter by dwells by accumulating through")
    old_len = len(dw.data)

    flt_cols = params["filter_cols"]
    is_critical = dw.data[flt_cols["refresh"]] | dw.data[flt_cols["crit"]]
    is_long_enough = dw.data[params["dwell_time_col"]] >= 0
    mask_col = "keep_dwells"
    dw.data[mask_col] = is_critical & is_long_enough

    accum_cols_internal = [dw.trip_dist, dw.trip_dur, dw.reset]
    accum_cols_fw = accum_cols_internal + params["accum_cols_forward_extra"]
    accum_cols_rv = params["accum_cols_reverse"]
    accum_cols = accum_cols_fw + accum_cols_rv
    revs = ([False] * len(accum_cols_fw)) + ([True] * len(accum_cols_rv))
    dw.accum_masked(mask_col, accum_cols=accum_cols, reverse=revs, inplace=True)

    dw.data[mask_col] = dw.data[mask_col].astype("boolean")
    dw.data[mask_col] = dw.data[mask_col].replace(False, pd.NA)
    dw.data.dropna(subset=mask_col, inplace=True)
    dw.data[mask_col] = dw.data[mask_col].astype(bool)
    drop_cols = [mask_col] + params["drop_cols"] + accum_cols
    dw.data = dw.data.drop(columns=drop_cols)
    renamer = {f"{old}_{mask_col}": old for old in accum_cols}
    dw.data = dw.data.rename(columns=renamer)
    dw.data[dw.reset] = dw.data[dw.reset].astype(bool)

    new_len = len(dw.data)
    abs_diff = old_len - new_len
    pct_diff = round(abs_diff / old_len * 100, 1)
    logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")
    return dw


def mark_shift_powers(dw: DwellSet, params: dict) -> DwellSet:
    """Mark the maximum available power in the remainder of the shift."""
    refr_col = params["refresh_col"]
    max_pow_col = params["max_power_col"]
    dw.accum_masked(
        keep_mask_col=refr_col,
        accum_cols=max_pow_col,
        reverse=True,
        agg_func=CumAggFunc.MAX,
        write_all=True,
        inplace=True,
    )
    acc_col = f"{max_pow_col}_{refr_col}"
    dw.data.loc[dw.data[refr_col], acc_col] = params["final_value"]
    dw.data[params["max_power_col_shift"]] = dw.data.groupby(dw.veh)[acc_col].shift(
        -1, fill_value=params["fill_value"]
    )
    dw.data.drop(columns=acc_col, inplace=True)
    return dw


def simulate_charging_choice(
    dw: DwellSet, vehs: pd.DataFrame, modes: pd.DataFrame, params: dict
) -> DwellSet:
    """Simulate the charging choices of each vehicle."""
    dw.sort_by_veh_time()
    strat = ForwardLookingChargingChoiceStrategy(**params["input_cols"])
    dw.data = strat.run(dwells=dw, vehs=vehs, modes=modes)
    dw.data = dw.data.drop(columns=params["drop_cols"])
    return dw


def apply_delays(dw: DwellSet, params: dict) -> DwellSet:
    """Apply the delays found in charging choice to dwell duration, start time, and end
    time.

    We apply the cumulative delay up to the present time to the beginning and end times
    of each dwell. Then we additionally reduce the dwell period by the delay reduction
    and increase the end time by the new delay added at this dwell.
    """
    dly_cols = params["delay_columns"]
    tdelt_cols = {}
    for prm_key, col in dly_cols.items():
        td_col = f"{col}_tdelta"
        dw.data[td_col] = pd.to_timedelta(dw.data[col], unit=params["delay_unit"])
        tdelt_cols.update({prm_key: td_col})

    cum_dly = dw.data[tdelt_cols["cumul_hrs"]]
    dw.data[dw.start] += cum_dly
    dw.data[dly_cols["dwell_hrs"]] = total_hours(
        dw.data[tdelt_cols["dwell_hrs"]]
        - dw.data[tdelt_cols["decrease_hrs"]]
        + dw.data[tdelt_cols["increase_hrs"]]
    )
    dw.data[dw.end] += cum_dly + dw.data[tdelt_cols["increase_hrs"]]

    dw.data = dw.data.drop(columns=list(tdelt_cols.values()))
    return dw


def manage_charging(dw: DwellSet, params: dict) -> pd.DataFrame:
    """Manage the charging of vehicles within each dwell to create charging events."""
    # Drop dwells with NaN charging energy, which probably resulted from vehicle deaths
    dw.data = dw.data.dropna(subset=params["drop_na_cols"])

    # Manage charging energy into power
    manager_cls = _MANAGER_MAP[params["charging_manager"]]
    manager = manager_cls(dw=dw, **params["input_cols"])
    events = manager.get_events()
    pow_col = manager_cls.suffixes["power"]
    events[pow_col] = events[pow_col].round(params["round_decimals"])
    return events
