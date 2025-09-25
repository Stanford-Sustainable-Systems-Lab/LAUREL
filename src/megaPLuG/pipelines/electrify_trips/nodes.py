"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd

from megaPLuG.models.charging_algorithms import ForwardLookingChargingChoiceStrategy
from megaPLuG.models.dwell_sets import CumAggFunc, DwellSet
from megaPLuG.utils.data import merge_dataframes_node
from megaPLuG.utils.mode_masks import bool_arr_to_bits
from megaPLuG.utils.params import build_df_from_dict
from megaPLuG.utils.time import total_hours

logger = logging.getLogger(__name__)


def filter_vehicles(dw: DwellSet, vehs: pd.DataFrame) -> DwellSet:
    """Filter out vehicles that we're not considering."""
    logger.info("Filter by vehicles by direct dropping")
    if not dw.is_dask:
        old_len = len(dw.data)

    if not dw.is_dask:
        keep_idx = np.intersect1d(dw.data.index.values, vehs.index.values)
        dw.data = dw.data.loc[keep_idx]
    else:
        veh_idx = vehs.index.to_frame(index=False)
        dw.data = dw.data.merge(
            veh_idx, left_index=True, right_on=vehs.index.name, how="inner"
        )
        dw.data = dw.data.drop(columns=[vehs.index.name])

    if not dw.is_dask:
        num_no_dwell_vehs = np.setdiff1d(vehs.index.values, keep_idx).size
        if num_no_dwell_vehs > 0:
            logger.warning(
                f"{num_no_dwell_vehs} vehicles were not found in the dwell data."
            )

        new_len = len(dw.data)
        abs_diff = old_len - new_len
        pct_diff = round(abs_diff / old_len * 100, 1)
        logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")
    return dw


def calc_dwell_durations(dw: DwellSet, params: dict) -> DwellSet:
    """Mark dwells which could provide substantial SoC to each vehicle."""
    iocols = params["in_out_time_cols"]
    iounit = params["in_out_time_unit"]
    for col in iocols.values():
        if dw.is_dask:
            dw.data[col] = dd.to_timedelta(dw.data[col], unit=iounit)
        else:
            dw.data[col] = pd.to_timedelta(dw.data[col], unit=iounit)

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
    modes_copy = dict(modes)  # avoid mutating input params dict
    name_col = modes_copy.pop("name_column")
    id_col = modes_copy.pop("id_column")
    modes_df = pd.DataFrame.from_dict(data=modes_copy, orient="index")
    modes_df.index.name = name_col
    modes_df = modes_df.reset_index()
    modes_df.index.name = id_col
    return modes_df


def prepare_mode_loc_corresp(modes: pd.DataFrame, params: dict) -> DwellSet:
    """Assign modes to each dwell using a boolean vector of mode availability."""
    avail_dict = params["charge_modes_avail"]
    avails = build_df_from_dict(
        d=avail_dict["values"],
        id_cols=avail_dict["id_columns"],
        value_col=avail_dict["value_col"],
    )

    poss = modes["mode_name"].to_numpy()
    max_power_source = params["max_power_source_col"]
    power_arr = modes[max_power_source].to_numpy()
    # Build boolean availability matrix (rows: locations, cols: modes)
    # Using vectorized comprehension via list for clarity then stack (acceptable size)
    bool_rows = []
    for av in avails[avail_dict["value_col"]].values:
        bool_rows.append(np.isin(poss, av))
    if bool_rows:
        bool_arr = np.vstack(bool_rows)
    else:
        bool_arr = np.zeros((0, poss.shape[0]), dtype=bool)

    # Build bitmask column via njit function
    bitmask_col = params["value_col_bool"]  # reuse existing param name
    avails[bitmask_col] = bool_arr_to_bits(bool_arr)

    # Max power derivation: elementwise multiply then row max
    if bool_arr.shape[0]:
        avails[params["max_power_col"]] = (bool_arr * power_arr[None, :]).max(axis=1)
    else:
        avails[params["max_power_col"]] = np.array([], dtype=float)

    # Location categorical treatment (unchanged)
    avails[params["loc_col"]] = pd.Categorical(avails[params["loc_col"]])
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
    nrg_col_cur = params["energy_col"]

    kws = {}
    if dw.is_dask:
        kws.update({"meta": ("x", "f8")})
    # Using no "fill_value" in shift is okay here because the column is NaN-able (float)
    dw.data[nrg_col_next] = dw.data.groupby(dw.veh)[nrg_col_cur].shift(-1, **kws)
    dw.data[nrg_col_next] = dw.data.groupby(dw.veh)[nrg_col_next].ffill()

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
    if dw.is_dask:
        dw.data = dw.data.drop(columns=[crit_bnd_col])
    else:
        dw.data.drop(columns=[crit_bnd_col], inplace=True)

    return dw


def filter_dwells(dw: DwellSet, params: dict) -> DwellSet:
    """Filter out the en-route dwells on non-critical days.

    We drop dwells wich are:
        - not a refresh location OR not enroute during a critical day
        - AND which have a duration longer than the plug_in + plug_out time
    """
    logger.info("Filter by dwells by accumulating through")
    if not dw.is_dask:
        old_len = len(dw.data)

    is_long_enough = dw.data[params["dwell_time_col"]] >= 0
    if params["filter_critical_days"]:
        flt_cols = params["filter_cols"]
        is_critical = dw.data[flt_cols["refresh"]] | dw.data[flt_cols["crit"]]
        mask_ser = is_long_enough & is_critical
    else:
        mask_ser = is_long_enough

    mask_col = "keep_dwells"
    dw.data[mask_col] = mask_ser

    accum_cols_internal = [dw.trip_dist, dw.trip_dur, dw.reset]
    accum_cols_fw = accum_cols_internal + params["accum_cols_forward_extra"]
    accum_cols_rv = params["accum_cols_reverse"]
    accum_cols = accum_cols_fw + accum_cols_rv
    revs = ([False] * len(accum_cols_fw)) + ([True] * len(accum_cols_rv))
    dw.accum_masked(mask_col, accum_cols=accum_cols, reverse=revs, inplace=True)

    dw.data[mask_col] = dw.data[mask_col].astype("boolean")
    dw.data[mask_col] = dw.data[mask_col].replace(False, pd.NA)
    if dw.is_dask:
        dw.data = dw.data.dropna(subset=mask_col)
    else:
        dw.data.dropna(subset=mask_col, inplace=True)
    dw.data[mask_col] = dw.data[mask_col].astype(bool)
    drop_cols = [mask_col] + params["drop_cols"] + accum_cols
    dw.data = dw.data.drop(columns=drop_cols)
    renamer = {f"{old}_{mask_col}": old for old in accum_cols}
    dw.data = dw.data.rename(columns=renamer)
    dw.data[dw.reset] = dw.data[dw.reset].astype(bool)

    if not dw.is_dask:
        new_len = len(dw.data)
        abs_diff = old_len - new_len
        pct_diff = round(abs_diff / old_len * 100, 1)
        logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")
    else:
        logger.info("Rows dropped: not calculated for Dask-backed DwellSets.")
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
    dw.data[acc_col] = dw.data[acc_col].where(
        ~dw.data[refr_col], other=params["final_value"]
    )

    kws = {"fill_value": params["fill_value"]}
    if dw.is_dask:
        kws.update({"meta": ("x", "f8")})

    shift_col = params["max_power_col_shift"]
    dw.data[shift_col] = dw.data.groupby(dw.veh)[acc_col].shift(-1, **kws)

    if dw.is_dask:
        dw.data = dw.data.drop(columns=acc_col)
    else:
        dw.data.drop(columns=acc_col, inplace=True)
    return dw


def simulate_charging_choice(
    dw: DwellSet, vehs: pd.DataFrame, modes: pd.DataFrame, params: dict
) -> DwellSet:
    """Simulate the charging choices of each vehicle."""
    if dw.is_dask:
        logger.info(
            "For Dask-backed DwellSets, we assume that sorting has been preserved."
        )
    else:
        dw.sort_by_veh_time()
    strat = ForwardLookingChargingChoiceStrategy(**params["input_cols"])

    if params["precompile"]:  # Note: Pre-compilation does not help distributed workers
        logger.info("Pre-compiling charging choice JIT-compiled functions.")
        try:
            dw_mock = dw.copy_without_data()
            base_df = dw.data._meta_nonempty if dw.is_dask else dw.data.iloc[:5].copy()
            dw_mock.data = base_df.copy()
            # Set all mode bits to 1 for each mock dwell
            all_mask = bool_arr_to_bits(np.ones(shape=(len(modes),), dtype=np.bool_))
            modes_avail_col = params["input_cols"]["modes_avail"]
            if modes_avail_col in dw_mock.data.columns:
                dw_mock.data[modes_avail_col] = np.repeat(
                    all_mask, repeats=len(dw_mock.data)
                )
            vehs_mock = vehs.iloc[: len(dw_mock.data)].copy()
            vehs_mock.index = dw_mock.data.index
            _ = strat.run(
                dwells=dw_mock, vehs=vehs_mock, modes=modes, show_progress=False
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Precompile attempt failed; continuing without: {e}")

    logger.info("Run charging choice simulation.")
    if dw.is_dask:

        def _run_charging_on_partition(partition_df: pd.DataFrame, dw_empty: DwellSet):
            # Create a temporary DwellSet for this partition
            dw_part = dw_empty.copy_without_data()
            dw_part.data = partition_df
            result = strat.run(
                dwells=dw_part, vehs=vehs, modes=modes, show_progress=False
            )
            return result

        # Create meta DataFrame to define output structure using schema generation
        meta = strat.get_output_schema(input=dw.data)
        dw.data = dw.data.map_partitions(
            _run_charging_on_partition, dw_empty=dw.copy_without_data(), meta=meta
        )
    else:
        dw.data = strat.run(dwells=dw, vehs=vehs, modes=modes)

    dw.data = dw.data.drop(columns=params["drop_cols"])
    return dw
