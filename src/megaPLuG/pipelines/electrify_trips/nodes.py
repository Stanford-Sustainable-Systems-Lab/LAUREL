"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

import logging

import numpy as np
import pandas as pd
from dask.dataframe.utils import make_meta
from tqdm import tqdm

from megaPLuG.models.charging_algorithms import charge_soc_thresh
from megaPLuG.models.dwell_sets import DwellSet
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
    if not dw.is_dask:
        logger.info("Filter by vehicles by direct dropping")
        old_len = len(dw.data)

    keep_idx = vehs.loc[vehs["has_home_base"], :].index.values
    dw.data = dw.data.loc[keep_idx, :]

    if not dw.is_dask:
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
    dw.data[refr_col] = dw.data[params["loc_col"]].isin(params["refresh_locations"])
    dw.accum_masked(refr_col, inplace=True)

    # Apply vehicle-specific estimated range
    vehs["range_estim"] = vehs[params["batt_cap_col"]] / vehs[params["consump_col"]]
    dw.data = dw.data.merge(vehs.loc[:, ["range_estim"]], how="left", on=dw.veh)

    crcol = params["crit_col"]
    veh_day_trip_dist_col = f"{dw.trip_dist}_{refr_col}"
    dw.data[crcol] = dw.data[veh_day_trip_dist_col] > dw.data["range_estim"]

    # Assume a critical day for partial days, since the point of the critical days
    # assumption is to reduce public charging on days when we're sure it's unnecessary
    dw.data[crcol] = dw.data[crcol].astype("boolean")
    dw.data.loc[~dw.data[refr_col], crcol] = pd.NA
    dw.data[crcol] = dw.data[crcol].groupby(dw.veh).bfill().fillna(True).astype(bool)
    return dw


def filter_dwells(dw: DwellSet, params: dict) -> DwellSet:
    """Filter out the en-route dwells on non-critical days."""
    if not dw.is_dask:
        logger.info("Filter by dwells by accumulating through")
        old_len = len(dw.data)

    flt_cols = params["filter_cols"]
    dw.data["keep_dwells"] = dw.data[flt_cols["substantial_soc"]] & (
        dw.data[flt_cols["refresh"]] | dw.data[flt_cols["crit"]]
    )
    dw.accum_masked("keep_dwells", inplace=True)

    dw.data["keep_dwells"] = dw.data["keep_dwells"].astype("boolean")
    dw.data["keep_dwells"] = dw.data["keep_dwells"].replace(False, pd.NA)
    if dw.is_dask:
        dw.data.dropna(subset="keep_dwells")
    else:
        dw.data.dropna(subset="keep_dwells", inplace=True)
    drop_cols = ["keep_dwells"] + list(flt_cols.values()) + params["drop_cols"]
    dw.data = dw.data.drop(columns=drop_cols)

    if not dw.is_dask:
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
    # TODO: It may be important to check for groupby monotonic increasing.
    icols = params["input_cols"]
    for col in params["output_cols"]:
        dw.data[col] = np.NaN  # Allocate columns to fill in, which avoids merging

    # Set arguments
    kws = {
        "func": charge_soc_thresh,
        "consumed_kwh_col": icols["consumed_kwh"],
        "avail_kw_col": icols["avail_kw"],
        "dwell_hrs_col": icols["dwell_hrs"],
        "reset_col": dw.reset,
        "veh_params": vehs,
        "out_cols": params["output_cols"],
    }

    # Run simulation
    if dw.is_dask:
        kws.update({"meta": make_meta(dw.data)})
        dw.data = dw.data.groupby(dw.veh, group_keys=False).apply(**kws)
    else:
        tqdm.pandas()
        dw.data = dw.data.groupby(dw.veh, group_keys=False).progress_apply(**kws)
    return dw


def summarize_vehicles(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Summarize the results for each vehicle."""
    dw.data["is_death"] = dw.data[params["dead_energy_col"]] < 0
    n_deaths = dw.data.groupby(dw.veh)["is_death"].sum()
    n_deaths.name = "n_deaths"
    vehs = vehs.merge(n_deaths, how="inner", on=dw.veh)

    logger.info("Deaths per vehicle:")
    logger.info(n_deaths.describe())
    return vehs


def get_hex_events_from_dwells(dw: DwellSet, params: dict) -> pd.DataFrame:
    """Convert vehicle dwells to hexagon events."""
    hex_kw_cols = [f"{seqn}_hex_kw_diff" for seqn in params["seq_names"]]
    dw.data[hex_kw_cols[0]] = dw.data["charge_kwh"] / dw.data["dwell_time_hrs"]
    dw.data[hex_kw_cols[1]] = -dw.data[hex_kw_cols[0]]

    dw.data = dw.data.dropna(subset=hex_kw_cols)
    dw.seq_names = params["seq_names"]
    events = dw.to_hex_profiles()
    return events
