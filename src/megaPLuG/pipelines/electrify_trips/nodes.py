"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

import logging

import numpy as np
import pandas as pd

from megaPLuG.models.charging_algorithms import SoCThreshChargingChoiceStrategy
from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.models.manage_charging import _MANAGER_MAP
from megaPLuG.utils.params import build_df_from_dict, flatten_dict
from megaPLuG.utils.time import total_hours

logger = logging.getLogger(__name__)


def set_vehicle_params(vehs: pd.DataFrame, params: dict) -> DwellSet:
    """Set vehicle parameters in advance of simulation."""
    # Seed is based on master seed and vehicle's id to ensure that vehicles are
    # individually controllable without impacting all other vehicles.
    orig_idx = vehs.index.names
    vehs = vehs.reset_index()

    for k, v in params.items():
        if isinstance(v, dict) and set(v.keys()) == {"id_columns", "values"}:
            # If this is a merge-type param, sensitive to already-defined parameters
            par_df = build_df_from_dict(
                d=v["values"], id_cols=v["id_columns"], value_col=k
            )
            vehs = vehs.merge(par_df, how="left", on=v["id_columns"], indicator="_mrg")
            if np.any(vehs["_mrg"] == "left_only"):
                raise RuntimeError(
                    f"Parameter values for {k} do not cover all vehicles."
                )
            else:
                vehs = vehs.drop(columns=["_mrg"])
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
    logger.info("Filter by vehicles by direct dropping")
    old_len = len(dw.data)

    keep_idx = vehs.loc[vehs["has_home_base"], :].index.values
    dw.data = dw.data.loc[keep_idx, :]

    new_len = len(dw.data)
    abs_diff = old_len - new_len
    pct_diff = round(new_len / old_len * 100, 1)
    logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")
    return dw


def set_charging_availability(dw: DwellSet, locs: dict) -> DwellSet:
    """Set the charging availability for each session."""
    dw.data["max_power_kw"] = locs["charging_rate_kw"]
    return dw


def mark_substantial_dwells(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> DwellSet:
    """Mark dwells which could provide substantial SoC to each vehicle."""
    pcols = params["veh_param_cols"]
    for col, unit in params["time_cols_w_unit"].items():
        vehs[col] = pd.to_timedelta(vehs[col], unit=unit)
    dw.data = dw.data.merge(vehs.loc[:, list(pcols.values())], how="left", on=dw.veh)

    # Adjust dwell start and end times to allow time for vehicle to plug in and out
    dw.data[dw.end] = dw.data[dw.end] - dw.data[pcols["plug_out"]]
    dw.data[dw.start] = dw.data[dw.start] + dw.data[pcols["plug_in"]]

    dwell_time_col = params["dwell_time_col"]
    dw.data[dwell_time_col] = total_hours(dw.data[dw.end] - dw.data[dw.start])

    # Calculate potential SoC increase of this dwell
    dw.data["soc_boost_potential"] = (
        dw.data[params["max_power_col"]]
        * dw.data[dwell_time_col]
        / dw.data[pcols["batt_cap"]]
    )
    dw.data["big_boost"] = dw.data["soc_boost_potential"] > dw.data[pcols["soc_boost"]]

    drop_cols = list(pcols.values()) + ["soc_boost_potential"]
    dw.data = dw.data.drop(columns=drop_cols)
    return dw


def mark_critical_days(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> DwellSet:
    """Mark critical days, vehicle-days which cannot be achieved on a single charge."""
    refr_col = params["refresh_col"]
    refr_locs = set(params["refresh_locations"])
    dw.data[refr_col] = dw.data[params["loc_col"]].isin(refr_locs)
    dw.accum_masked(refr_col, accum_cols=dw.trip_dist, inplace=True)

    # Apply vehicle-specific estimated range
    vehs["range_estim"] = vehs[params["batt_cap_col"]] / vehs[params["consump_col"]]
    dw.data = dw.data.merge(vehs.loc[:, ["range_estim"]], how="left", on=dw.veh)

    crcol = params["crit_col"]
    veh_day_trip_dist_col = f"{dw.trip_dist}_{refr_col}"
    dw.data[crcol] = dw.data[veh_day_trip_dist_col] > dw.data["range_estim"]

    # Assume a critical day for partial days, since the point of the critical days
    # assumption is to reduce public charging on days when we're sure it's unnecessary
    # Note that "boolean" is different from "bool" data type. See Pandas BooleanArray
    is_critical = dw.data[crcol].astype("boolean")
    is_refresh = dw.data[refr_col].astype("boolean")
    dw.data[crcol] = (is_refresh & is_critical) ^ ~(is_refresh | pd.NA)
    dw.data[crcol] = (
        dw.data[crcol].groupby(dw.veh, sort=False).bfill().fillna(True).astype(bool)
    )
    return dw


def filter_dwells(dw: DwellSet, params: dict) -> DwellSet:
    """Filter out the en-route dwells on non-critical days."""
    logger.info("Filter by dwells by accumulating through")
    old_len = len(dw.data)

    flt_cols = params["filter_cols"]
    dw.data["keep_dwells"] = dw.data[flt_cols["substantial_soc"]] & (
        dw.data[flt_cols["refresh"]] | dw.data[flt_cols["crit"]]
    )
    accum_cols = [dw.trip_dist, dw.trip_dur, dw.reset]
    dw.accum_masked("keep_dwells", accum_cols=accum_cols, inplace=True)

    dw.data["keep_dwells"] = dw.data["keep_dwells"].astype("boolean")
    dw.data["keep_dwells"] = dw.data["keep_dwells"].replace(False, pd.NA)
    dw.data.dropna(subset="keep_dwells", inplace=True)
    drop_cols = ["keep_dwells"] + list(flt_cols.values()) + params["drop_cols"]
    dw.data = dw.data.drop(columns=drop_cols)

    new_len = len(dw.data)
    abs_diff = old_len - new_len
    pct_diff = round(new_len / old_len * 100, 1)
    logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")
    return dw


def calc_energy_use(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> DwellSet:
    """Calculate energy use for all trips."""
    dw.data = dw.data.merge(vehs.loc[:, [params["consump_col"]]], how="left", on=dw.veh)
    dw.data[params["energy_col"]] = (
        dw.data[dw.trip_dist] * dw.data[params["consump_col"]]
    )
    dw.data = dw.data.drop(columns=[params["consump_col"]])
    return dw


def simulate_charging_choice(
    dw: DwellSet, vehs: pd.DataFrame, params: dict
) -> DwellSet:
    """Simulate the charging choices of each vehicle."""
    dw.sort_by_veh_time()
    strat = SoCThreshChargingChoiceStrategy(**params["input_cols"])
    dw.data = strat.run(dwells=dw, vehs=vehs)
    return dw


def assign_regions(dw: DwellSet, hex_regions: pd.DataFrame) -> DwellSet:
    """Assign larger regions to the DwellSet based on hexagon ids."""
    orig_idx = dw.data.index.names
    dw.data = dw.data.reset_index()
    dw.data = dw.data.merge(hex_regions, how="left", on=dw.hex)
    dw.data = dw.data.set_index(orig_idx)
    return dw


def assign_scale_up_factor(
    dw: DwellSet, vehs: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Assign the factor by which each dwell's power will be scaled up."""
    if params["apply_scaling"]:
        mrg = vehs.loc[:, params["veh_cols"]]
        dw.data = dw.data.merge(mrg, how="left", on=dw.veh)
    else:
        dw.data.loc[:, params["veh_cols"]] = 1.0
    return dw


def manage_charging(dw: DwellSet, params: dict) -> pd.DataFrame:
    """Manage the charging of vehicles within each dwell to create charging events."""
    # Drop dwells with NaN charging energy, which probably resulted from vehicle deaths
    dw.data = dw.data.dropna(subset=params["drop_na_cols"])

    # Manage charging energy into power
    manager_cls = _MANAGER_MAP[params["charging_manager"]]
    manager = manager_cls(dw=dw, **params["input_cols"])
    events = manager.get_events()
    return events
