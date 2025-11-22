"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""

import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd

from megaplug.models.charging_algorithms import ForwardLookingChargingChoiceStrategy
from megaplug.models.dwell_sets import CumAggFunc, DwellSet
from megaplug.utils.data import merge_dataframes_node
from megaplug.utils.mode_masks import bool_arr_to_bits
from megaplug.utils.time import total_hours

logger = logging.getLogger(__name__)


def filter_vehicles(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> DwellSet:
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

    if dw.is_dask:
        dw.data = dw.data.repartition(npartitions=params["n_partitions"])

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


def assign_modes(dw: DwellSet, modes: pd.DataFrame, params: dict) -> DwellSet:  # noqa: PLR0912
    """Assign charging modes to each dwell."""
    all_modes = modes[params["mode_col"]].to_list()
    modes_in_use = [mode for mode in all_modes if mode in dw.data.columns]
    if len(modes_in_use) > 0:
        raise ValueError(f"Mode columns already in use: {modes_in_use}")

    # Build one boolean column for each location-only mode
    mode_locs = params["loc_based_mode_avail"]
    for mode, selector in mode_locs.items():
        if mode not in all_modes:
            raise ValueError(f"Charging mode '{mode}' not available.")
        if isinstance(selector, bool):
            dw.data[mode] = selector
        elif isinstance(selector, dict):
            locs = selector["loc_groups"]
            invert = selector.get("invert_selection", False)
            dw.data[mode] = dw.data[params["loc_group_col"]].isin(locs)
            if invert:
                dw.data[mode] = ~dw.data[mode]
        else:
            raise ValueError(
                "Available modes must be boolean or dict of location groups."
            )

    # Add a boolean column for the vehicle-based depot mode
    mode_vehs = params["veh_based_mode_avail"]
    vmode_col = mode_vehs["mode_name"]
    if vmode_col not in dw.data.columns:
        dw.data[vmode_col] = True

    orig_cols = list(dw.data.columns)
    dw.data[vmode_col] &= dw.data[mode_vehs["ratio_col"]] > mode_vehs["ratio_thresh"]
    if dw.is_dask:  # Reset column ordering to keep in sync with the Dask Dataframe meta
        dw.data = dw.data.map_partitions(
            lambda part: part.loc[:, orig_cols], meta=dw.data._meta
        )

    # Check that all modes are accounted for
    modes_missed = [mode for mode in all_modes if mode not in dw.data.columns]
    if len(modes_missed) > 0:
        raise ValueError(f"The following charging modes are unassigned: {modes_missed}")

    # Build mode bitmask column
    if dw.is_dask:
        all_mask = bool_arr_to_bits(np.ones(shape=(len(modes),), dtype=np.bool_))
        meta = dw.data._meta.assign(mode_mask_bits=all_mask.dtype)
        dw.data = dw.data.map_partitions(
            lambda part: part.assign(
                mode_mask_bits=bool_arr_to_bits(
                    part[all_modes].to_numpy(dtype=np.bool_)
                )
            ),
            meta=meta,
        )
    else:
        dw.data["mode_mask_bits"] = bool_arr_to_bits(
            dw.data[all_modes].to_numpy(dtype=np.bool_)
        )
    dw.data = dw.data.rename(columns={"mode_mask_bits": params["mode_mask_col"]})

    # Build maximum power column
    power_mapper = build_mode_power_lut(
        mode_names=modes[params["mode_col"]],
        mode_powers=modes[params["max_power_source_col"]],
    )
    if dw.is_dask:
        meta = (params["max_power_col"], modes[params["max_power_source_col"]].dtype)
        dw.data[params["max_power_col"]] = dw.data[params["mode_mask_col"]].map(
            power_mapper, meta=meta
        )
    else:
        dw.data[params["max_power_col"]] = dw.data[params["mode_mask_col"]].map(
            power_mapper
        )

    # Drop boolean columns
    dw.data = dw.data.drop(columns=all_modes)

    return dw


def build_mode_power_lut(
    mode_names: pd.Series, mode_powers: pd.Series
) -> dict[int, float]:
    """Map every mode availability bitmask to its maximum deliverable power."""
    name_ls = mode_names.tolist()
    power_arr = mode_powers.to_numpy(dtype=float)

    combos = pd.DataFrame(
        np.array(np.meshgrid(*[[False, True]] * len(name_ls))).T.reshape(
            -1, len(name_ls)
        ),
        columns=name_ls,
        dtype=bool,
    )

    bitmask_col = "mode_mask_bits"
    combos[bitmask_col] = bool_arr_to_bits(combos[name_ls].to_numpy(dtype=np.bool_))
    combos["max_power"] = (
        combos[name_ls].to_numpy(dtype=float) * power_arr[None, :]
    ).max(axis=1)

    res_ser = pd.Series(
        combos["max_power"].to_numpy(dtype=float),
        index=combos[bitmask_col].to_numpy(),
        dtype=float,
    )
    res_dict = res_ser.to_dict()

    return res_dict


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


def _filter_dwells_core(
    dw: DwellSet,
    keep_col: str,
    accum_cols_fw_extra: list[str] | None = None,
    accum_cols_rv: list[str] | None = None,
    drop_cols_extra: list[str] | None = None,
) -> DwellSet:
    """Core dwell filtering functionality, used several times.

    Wraps DwellSet.accum_masked() with additional column renaming and setup and teardown.
    """
    accum_cols = [dw.trip_dist, dw.trip_dur, dw.reset]
    revs = [False] * len(accum_cols)

    if accum_cols_fw_extra is not None:
        accum_cols += accum_cols_fw_extra
        revs += [False] * len(accum_cols_fw_extra)

    if accum_cols_rv is not None:
        accum_cols += accum_cols_rv
        revs += [True] * len(accum_cols_rv)

    dw.accum_masked(keep_col, accum_cols=accum_cols, reverse=revs, inplace=True)

    dw.data[keep_col] = dw.data[keep_col].astype("boolean")
    dw.data[keep_col] = dw.data[keep_col].replace(False, pd.NA)
    if dw.is_dask:
        dw.data = dw.data.dropna(subset=keep_col)
    else:
        dw.data = dw.data.dropna(subset=keep_col)
    dw.data[keep_col] = dw.data[keep_col].astype(bool)
    drop_cols = [keep_col] + accum_cols
    if drop_cols_extra is not None:
        drop_cols.extend(drop_cols_extra)
    dw.data = dw.data.drop(columns=drop_cols)
    renamer = {f"{old}_{keep_col}": old for old in accum_cols}
    dw.data = dw.data.rename(columns=renamer)
    dw.data[dw.reset] = dw.data[dw.reset].astype(bool)
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

    is_long_enough = (
        dw.data[params["dwell_time_col"]] >= 0
    )  # Note, this duration already takes into account the plug in/out times
    if params["filter_critical_days"]:
        flt_cols = params["filter_cols"]
        is_critical = dw.data[flt_cols["refresh"]] | dw.data[flt_cols["crit"]]
        is_optional = dw.data[dw.end] <= dw.data[dw.start]
        keep_ser = is_long_enough & (is_critical | ~is_optional)
    else:
        keep_ser = is_long_enough

    dw.data["keep_dwells"] = keep_ser

    dw = _filter_dwells_core(
        dw=dw,
        keep_col="keep_dwells",
        accum_cols_fw_extra=params["accum_cols_forward_extra"],
        accum_cols_rv=params["accum_cols_reverse"],
        drop_cols_extra=params["drop_cols"],
    )

    if not dw.is_dask:
        new_len = len(dw.data)
        abs_diff = old_len - new_len
        pct_diff = round(abs_diff / old_len * 100, 1)
        logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")
    else:
        logger.info("Rows dropped: not calculated for Dask-backed DwellSets.")
    return dw


def mark_shift_powers(dw: DwellSet, params: dict) -> DwellSet:
    """Mark the maximum available power in the remainder of the shift.

    This node begins to enforce the auto-skipping of charging at dwells which are
    neither refreshes nor critical. We need to keep these dwells in the pipeline for
    use in the scaling probabilities later, but we want to skip even considering
    charging at them for the sake of computational efficiency.
    """
    refr_col = params["refresh_col"]
    crit_col = params["crit_col"]
    max_pow_col = params["max_power_col"]
    dont_auto_skip = dw.data[crit_col] | dw.data[refr_col]
    dw.data["power_w_skips"] = dw.data[max_pow_col].where(dont_auto_skip, other=0.0)
    dw.accum_masked(
        keep_mask_col=refr_col,
        accum_cols="power_w_skips",
        reverse=True,
        agg_func=CumAggFunc.MAX,
        write_all=True,
        inplace=True,
    )
    acc_col = f"power_w_skips_{refr_col}"
    dw.data[acc_col] = dw.data[acc_col].where(
        ~dw.data[refr_col], other=params["final_value"]
    )

    kws = {"fill_value": params["fill_value"]}
    if dw.is_dask:
        kws.update({"meta": ("x", "f8")})

    shift_col = params["max_power_col_shift"]
    dw.data[shift_col] = dw.data.groupby(dw.veh)[acc_col].shift(-1, **kws)

    dw.data = dw.data.drop(columns=["power_w_skips", acc_col])
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
        _ = strat.run(dwells=dw_mock, vehs=vehs_mock, modes=modes, show_progress=False)

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


def filter_dwells_post(dw: DwellSet, params: dict) -> DwellSet:
    """Filter out the optional stops not taken.

    We drop dwells wich are:
        - optional stops (zero duration)
        - no charging (charge kwh is zero)
    """
    logger.info("Filter by dwells by accumulating through")
    if not dw.is_dask:
        old_len = len(dw.data)

    if params["filter_unused_optionals"]:
        is_optional = dw.data[dw.end] <= dw.data[dw.start]
        has_some_charge = dw.data[params["filter_cols"]["charge"]] > 0.0
        dw.data["keep_dwells"] = ~is_optional | (is_optional & has_some_charge)

        dw = _filter_dwells_core(
            dw=dw,
            keep_col="keep_dwells",
            accum_cols_fw_extra=params["accum_cols_forward_extra"],
            accum_cols_rv=params["accum_cols_reverse"],
            drop_cols_extra=params["drop_cols"],
        )
    else:
        pass

    if not dw.is_dask:
        new_len = len(dw.data)
        abs_diff = old_len - new_len
        pct_diff = round(abs_diff / old_len * 100, 1)
        logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")
    else:
        logger.info("Rows dropped: not calculated for Dask-backed DwellSets.")
    return dw
