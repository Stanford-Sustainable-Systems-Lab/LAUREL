"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import gc
import logging

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed
from tqdm.auto import tqdm

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.models.group_times import AdaptiveTimeGrouper, HourOfWeekdayGrouper
from megaPLuG.models.manage_charging import _MANAGER_MAP, ProfileType
from megaPLuG.models.probability_localization import (
    ElectProbLocalizer,
    ElectProbLocalizerConfig,
)
from megaPLuG.models.sampling import (
    build_entity_mask_array,
    discretize_sparse_profiles,
    normalize_sparse,
    sample_profiles,
)
from megaPLuG.models.summarize import IntervalBeginSpreader, NonzeroGroupedSummarizer
from megaPLuG.utils.data import IndexIntegerizer, filter_by_vals_in_cols
from megaPLuG.utils.h3 import cells_to_region_polygons
from megaPLuG.utils.logging import SuppressLogs
from megaPLuG.utils.time import (
    calc_local_time,
    calc_time_zones_from_hexes,
    total_hours,
    total_time_units,
)

logger = logging.getLogger(__name__)


# ruff: noqa: PLR0915
def summarize_vehicles(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Summarize the results for each vehicle."""
    # Set up arguments for describe calls
    qtls_descr = sorted(list(set(params["quantiles"]).difference([0.0, 1.0])))

    ## Build explanatory variables
    # Number of deaths by vehicle
    dw.data["hex_id_prev"] = dw.data.groupby(dw.veh)[dw.hex].shift(1, fill_value=0)
    dw.data["circle_trip"] = dw.data["hex_id_prev"] == dw.data[dw.hex]

    dw.data["is_death"] = dw.data[params["dead_energy_col"]] < 0
    dw.data["is_death_circle"] = dw.data["is_death"] & dw.data["circle_trip"]

    dw.data["is_dead"] = dw.data[params["dead_energy_col"]].isna() | dw.data["is_death"]
    dw.data["is_resusc"] = (
        dw.data["is_dead"] & ~dw.data[params["charge_energy_col"]].isna()
    )
    dw.data["dead_hrs"] = (
        dw.data["is_dead"] * dw.data[dw.trip_dur]
        + (dw.data["is_dead"] & ~dw.data["is_resusc"])
        * dw.data[params["dwell_dur_col"]]
    )

    # Shift identification
    dw.data["shift_id"] = dw.data.groupby(dw.veh)[params["refresh_col"]].cumsum()
    dw.data["shift_id"] = dw.data.groupby(dw.veh)["shift_id"].shift(1, fill_value=0)

    # Delay
    dw.data["delay_inc_hrs_shift"] = dw.data.groupby(dw.veh)[
        params["delay_inc_hrs_col"]
    ].shift(1, fill_value=0)

    ## Compute performance metrics by vehicle and shift
    shifts_summ = dw.data.groupby([dw.veh, "shift_id"]).agg(
        n_deaths_all=pd.NamedAgg("is_death", "sum"),
        n_deaths_circle=pd.NamedAgg("is_death_circle", "sum"),
        max_delay_hrs=pd.NamedAgg(params["delay_hrs_col"], "max"),
        prev_delay_hrs_inc=pd.NamedAgg("delay_inc_hrs_shift", "first"),
        init_delay_hrs=pd.NamedAgg(params["delay_hrs_col"], "first"),
        final_delay_hrs=pd.NamedAgg(params["delay_hrs_col"], "last"),
        trip_dur_sum=pd.NamedAgg(dw.trip_dur, "sum"),
        dwell_dur_sum=pd.NamedAgg(params["dwell_dur_col"], "sum"),
        dwell_dur_last=pd.NamedAgg(params["dwell_dur_col"], "last"),
    )

    # Deaths
    shifts_summ["n_deaths_addressable"] = (
        shifts_summ["n_deaths_all"] - shifts_summ["n_deaths_circle"]
    )
    shifts_summ["any_deaths_all"] = shifts_summ["n_deaths_all"] > 0
    shifts_summ["any_deaths_addressable"] = shifts_summ["n_deaths_addressable"] > 0

    # Delays
    scols = params["summary_cols"]
    shifts_summ = shifts_summ.rename(columns={"max_delay_hrs": scols["max_delay"]})
    # The delay incurred by this shift is: the cumulative delay level when the vehicle
    # arrives at the final depot MINUS the cumulative delay level when the vehicle arrives
    # at its first (non-refresh) stop in the shift PLUS the increase in delay incurred
    # at the refresh stop directly before this shift (in preparation for this shift)
    shifts_summ[scols["shift_delay"]] = (
        shifts_summ["final_delay_hrs"]
        - shifts_summ["init_delay_hrs"]
        + shifts_summ["prev_delay_hrs_inc"]
    )
    # Vehicle death can sometimes cause cumulative delay to drop, so we ignore those
    # shifts, setting their delay to zero
    shifts_summ[scols["shift_delay"]] = shifts_summ[scols["shift_delay"]].clip(
        lower=0.0
    )

    shifts_summ[scols["shift_dur"]] = (
        shifts_summ["trip_dur_sum"]
        + shifts_summ["dwell_dur_sum"]
        - shifts_summ["dwell_dur_last"]
    )
    shifts_summ[scols["shift_dur_delayed"]] = (
        shifts_summ[scols["shift_dur"]] + shifts_summ[scols["shift_delay"]]
    )

    max_dur = params["shift_max_dur_hrs"]
    shifts_summ[scols["shift_dur_delayed_thresh"]] = (
        shifts_summ[scols["shift_dur"]] <= max_dur
    ) & (shifts_summ[scols["shift_dur_delayed"]] > max_dur)
    shifts_summ[scols["delay_frac"]] = (
        shifts_summ[scols["shift_delay"]] / shifts_summ[scols["shift_dur"]]
    )

    # Compute performance metrics by vehicle
    shifts_summ = shifts_summ.reset_index("shift_id")
    vehs_summ_grp = shifts_summ.groupby(dw.veh, sort=False)

    vehs_summ_point = vehs_summ_grp.agg(
        n_shifts=pd.NamedAgg("shift_id", "count"),
        n_shifts_w_deaths=pd.NamedAgg("any_deaths_all", "sum"),
        n_shifts_w_deaths_addressable=pd.NamedAgg("any_deaths_addressable", "sum"),
        n_shifts_delayed_over_thresh=pd.NamedAgg(
            scols["shift_dur_delayed_thresh"], "sum"
        ),
    )
    vehs_summ_point["pct_shifts_w_deaths"] = (
        vehs_summ_point["n_shifts_w_deaths"] / vehs_summ_point["n_shifts"]
    ) * 100
    vehs_summ_point["pct_shifts_w_deaths_addressable"] = (
        vehs_summ_point["n_shifts_w_deaths_addressable"] / vehs_summ_point["n_shifts"]
    ) * 100
    vehs_summ_point["pct_shifts_delayed_past_thresh"] = (
        vehs_summ_point["n_shifts_delayed_over_thresh"] / vehs_summ_point["n_shifts"]
    ) * 100

    vehs_summ_dists = vehs_summ_grp[list(scols.values())].quantile(params["quantiles"])
    vehs_summ_dists = vehs_summ_dists.unstack(level=1)

    def build_col_name(val: str, pct: float) -> str:
        return f"{val}_{int(pct * 100)}"

    flat_cols = [build_col_name(name, pct) for name, pct in vehs_summ_dists.columns]
    vehs_summ_dists.columns = flat_cols

    vehs_summ = vehs_summ_point.merge(vehs_summ_dists, how="inner", on=dw.veh)
    vehs = vehs.merge(vehs_summ, how="inner", on=dw.veh)

    ## Report results by vehicle
    # Deaths
    logger.info("Percent of shifts with deaths (all causes) by vehicle:")
    logger.info(vehs["pct_shifts_w_deaths"].describe(percentiles=qtls_descr))

    logger.info("Percent of shifts with deaths (addressable) by vehicle:")
    logger.info(
        vehs["pct_shifts_w_deaths_addressable"].describe(percentiles=qtls_descr)
    )

    # Delays
    logger.info(
        f"Percent of shifts delayed past threshold time ({max_dur} hours) by vehicle:"
    )
    logger.info(vehs["pct_shifts_delayed_past_thresh"].describe(percentiles=qtls_descr))

    thresh_qtl = params["delay_frac_thresh_quantile"]
    report_pctl = int(thresh_qtl * 100)
    logger.info(
        f"Delay as fraction of vehicle shift length per vehicle [{report_pctl}th percentile]:"
    )
    report_col = build_col_name(scols["delay_frac"], thresh_qtl)
    logger.info(vehs[report_col].describe(percentiles=qtls_descr))

    thresh_qtl = params["max_delay_thresh_quantile"]
    report_pctl = int(thresh_qtl * 100)
    logger.info(
        f"Maximum absolute delay accumulated per vehicle [{report_pctl}th percentile]:"
    )
    report_col = build_col_name(scols["max_delay"], thresh_qtl)
    logger.info(vehs[report_col].describe(percentiles=qtls_descr))

    # Get boolean columns for which vehicles are included in load profiles
    thrs = params["thresholds"]
    vehs["dies_too_freq"] = (
        vehs["pct_shifts_w_deaths"] > thrs["pct_shifts_w_deaths_max"]
    )
    dftcol = build_col_name(scols["delay_frac"], thresh_qtl)
    vehs["delays_too_long_rel"] = vehs[dftcol] > thrs["delay_frac_max"]
    abstcol = build_col_name(scols["max_delay"], 1.00)
    vehs["delays_too_long_abs"] = vehs[abstcol] > thrs["delay_hrs_max"]
    vehs[params["drop_events_col"]] = (
        vehs["dies_too_freq"]
        | vehs["delays_too_long_rel"]
        | vehs["delays_too_long_abs"]
    )
    vehs[params["electrified_col"]] = ~vehs[params["drop_events_col"]]

    logger.info("Vehicles to be dropped [%] based on reason:")
    reasons = [
        "dies_too_freq",
        "delays_too_long_rel",
        "delays_too_long_abs",
        params["drop_events_col"],
    ]
    logger.info(np.round(vehs.loc[:, reasons].sum(axis=0) / len(vehs) * 100, 2))
    return vehs


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
        if dw.is_dask:
            dw.data[td_col] = dd.to_timedelta(dw.data[col], unit=params["delay_unit"])
        else:
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


def filter_dwells_pre_prob(dw: DwellSet, params: dict, pcols: dict) -> DwellSet:
    """Filter dwells down to only include the ones we want to summarize for probabilities.

    That is, we want the dwells which occur within:
        - The location target pool OR the location donor pool
        - The time spans expected (e.g. weekdays)

    DO NOT drop the zero-charge dwells. They're critical to the dwell-based sampling.
    Furthermore, I'll need to consider keeping the non-critical days around to retain
    these dwells.

    DO NOT drop the dwells from non-electrified vehicles. These will be essential for
    probability of electrification calculations.
    """
    if not dw.is_dask:
        old_len = len(dw.data)

    # Dwells which have at least some time within our time spans of interest
    if params["filter_out_weekends"]:
        FIRST_DAY_OF_WEEKEND = 5
        is_weekday = (dw.data[dw.start].dt.weekday < FIRST_DAY_OF_WEEKEND) | (
            dw.data[dw.end].dt.weekday < FIRST_DAY_OF_WEEKEND
        )
        dw.data = dw.data.loc[is_weekday, :]

    # Perform filtering
    if params["drop_na_cols"] is not None:
        dw.data = dw.data.dropna(subset=params["drop_na_cols"])

    if not dw.is_dask:
        new_len = len(dw.data)
        abs_diff = old_len - new_len
        pct_diff = round(abs_diff / old_len * 100, 1)
        logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")

    return dw


def filter_locs_pre_prob(locs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Filter locations to report upon."""
    grp_col = params["loc_group_col"]
    locs_report = locs.dropna(subset=grp_col)
    locs_report[grp_col] = locs_report[grp_col].astype(int)
    return locs_report


def get_unique_series(df: pd.DataFrame, col: str, dropna: bool = True) -> pd.Series:
    """Get the series from the df with the given name, and return only the unique values."""
    if col in df.columns:
        ser = df[col].sort_values()
    elif col in df.index.names:
        ser = df.index.get_level_values(col).to_series().sort_values()
    else:
        raise RuntimeError(
            f"Column {col} not found in the dataframe columns or index names."
        )
    uniq_ser = pd.Series(data=ser.unique(), name=col)
    if dropna:
        uniq_ser = uniq_ser.dropna()
    return uniq_ser


def build_class_frame(
    locs: pd.DataFrame, vehs: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Build the frame for all combinations of location and vehicle classes."""
    components = []
    for col in params["veh_class_cols"]:
        components.append(get_unique_series(vehs, col))
    for col in params["loc_class_cols"]:
        components.append(get_unique_series(locs, col))

    classes = pd.MultiIndex.from_product(components).to_frame(index=False)
    return classes


def compute_class_dwell_counts(
    classes: pd.DataFrame,
    dw: DwellSet,
    params: dict,
) -> pd.DataFrame:
    """Compute dwell counts for combinations of vehicle and location classes."""
    cls_cols = params["veh_class_cols"] + params["loc_class_cols"]
    n_obs_grper = cls_cols + [params["electrified_col"]]
    n_obs = dw.data.groupby(n_obs_grper, observed=True).size().sort_index()
    dwells_obs = n_obs.unstack(params["electrified_col"])
    renamer = {True: "n_dwells_electrified", False: "n_dwells_not_electrified"}
    dwells_obs.columns = [renamer[old] for old in dwells_obs.columns]
    dwells_obs["n_dwells_obs"] = dwells_obs.sum(axis=1)
    dwells_obs = dwells_obs.drop(columns=[renamer[False]])

    classes = classes.merge(dwells_obs, how="left", on=cls_cols)
    for col in ["n_dwells_obs", "n_dwells_electrified"]:
        classes[col] = classes[col].fillna(0).astype(int)

    return classes


def compute_known_adoption_totals(
    adopts: pd.DataFrame,
    params: dict,
    pcols: dict,
) -> pd.DataFrame:
    """Compute known adoption totals from forecasts."""
    adopts_sel = filter_by_vals_in_cols(adopts, params["filter_totals"])

    tot_col = params["totals_column"]
    adopts_sel.loc[:, tot_col] = adopts_sel[tot_col].astype(int)

    elect_col = pcols["electrified_col"]
    ftype_col = params["fuel_type_col"]
    adopts_sel[elect_col] = adopts_sel[ftype_col].isin(params["electrified_fuel_types"])
    adopt_grper = pcols["veh_class_cols"] + [elect_col]
    adopt_elect = adopts_sel.groupby(adopt_grper)[tot_col].sum()
    adopt_elect = adopt_elect.unstack(elect_col)
    renamer = {True: f"n_vehs_{elect_col}", False: f"n_vehs_not_{elect_col}"}
    adopt_elect.columns = [renamer[old] for old in adopt_elect.columns]

    adopt_elect[tot_col] = adopt_elect.sum(axis=1)
    adopt_elect = adopt_elect.drop(columns=[renamer[False]])
    return adopt_elect


def compute_dwell_rate(
    veh_classes: pd.DataFrame,
    dw: DwellSet,
    vehs: pd.DataFrame,
    params: dict,
    pcols: dict,
) -> pd.DataFrame:
    """Compute the rate of dwells per unit time."""
    dw_count_col = params["dwell_count_col"]
    obs_dur_col = params["obs_dur_col"]
    vclass_cols = pcols["veh_class_cols"]

    n_dwells = dw.data.groupby(dw.veh).size()
    n_dwells.name = dw_count_col
    n_dwells = n_dwells.to_frame()
    mrg = vehs.loc[:, [obs_dur_col] + vclass_cols]
    n_dwells = n_dwells.merge(mrg, how="right", left_index=True, right_index=True)

    t_unit = params["time_unit"]
    obs_t_units_col = f"obs_{t_unit}_units"
    n_dwells[obs_t_units_col] = total_time_units(n_dwells[obs_dur_col], unit=t_unit)
    dw_per_t_unit = n_dwells.groupby(vclass_cols, observed=True).agg(
        n_dwells=pd.NamedAgg(dw_count_col, "sum"),
        obs_units=pd.NamedAgg(obs_t_units_col, "sum"),
        n_vehs_obs=pd.NamedAgg(dw_count_col, "size"),
    )
    dw_per_t_unit = dw_per_t_unit.rename(
        columns={"obs_units": obs_t_units_col, "n_dwells": dw_count_col}
    )
    dw_per_t_unit[params["dwell_rate_col"]] = (
        dw_per_t_unit[dw_count_col] / dw_per_t_unit[obs_t_units_col]
    )

    veh_classes = veh_classes.merge(dw_per_t_unit, how="left", on=vclass_cols)
    return veh_classes


def compute_class_probs(
    cls: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Compute probabilities associated with location and vehicle cls.

    We compute:
        - P(L,V): the probability of a dwell falling into location class l and vehicle class v
        - P(L|V): the probability of a dwell falling into location class l given it is in vehicle class v
    """
    if len(params["loc_class_cols"]) > 1:
        raise ValueError(
            "Location class must be defined by a single column. If needed, combine columns together to achieve this."
        )
    if len(params["veh_class_cols"]) > 1:
        raise ValueError(
            "Vehicle class must be defined by a single column. If needed, combine columns together to achieve this."
        )

    vcols = params["veh_class_cols"]
    lcols = params["loc_class_cols"]
    pcols = params["columns"]

    # Get P(L|V) from sample
    dw_obs_col = pcols["n_dwells_obs"]
    dw_ele_col = pcols["n_dwells_obs_elect"]
    cls["n_dwells_obs_vclass"] = cls.groupby(vcols, observed=True)[
        dw_obs_col
    ].transform(lambda ser: ser.sum())
    cls["p_lclass_g_vclass"] = cls[dw_obs_col] / cls["n_dwells_obs_vclass"]

    # Get P(L,V) from sample
    cls["p_vclass_lclass"] = cls[dw_obs_col] / cls[dw_obs_col].sum()

    # Get P(E|V) from forecast
    tot_col = pcols["n_vehs_in_class"]
    veh_ele_col = pcols["n_vehs_electrified_in_class"]
    cls["p_elect_g_vclass"] = cls[veh_ele_col] / cls[tot_col]

    # Get consensus P(E|L,V) through data fusion
    cfg = ElectProbLocalizerConfig(
        loc_col=lcols[0],
        veh_col=vcols[0],
        n_electrified_col=dw_ele_col,
        n_obs_col=dw_obs_col,
    )
    fuser = ElectProbLocalizer(cls, config=cfg)
    cls["p_elect_g_vclass_lclass_cls"] = fuser.fit_transform()

    # Get number of dwells expected by class combination.
    dw_rate_col = pcols["dwell_rate"]
    tgt_veh_col = pcols["n_vehs_in_class"]
    cls["n_dwells_expected"] = (
        cls["p_lclass_g_vclass"] * cls[dw_rate_col] * cls[tgt_veh_col]
    )

    ## Compute output columns: unique values for each (loc class, veh class) pair
    # Get probability of observed-electrifiability: P(E,O|L,V)
    ocols = params["out_columns"]
    n_obs_dwells = cls[dw_obs_col].sum()
    cls[ocols["prob_obs_elect"]] = (
        cls["p_vclass_lclass"]
        * cls["p_elect_g_vclass_lclass_cls"]
        * n_obs_dwells
        / cls["n_dwells_expected"]
    )

    # Get number of expected electrified dwells per class PER location within class
    cls[ocols["n_elect_dwells_expected"]] = (
        cls["n_dwells_expected"] * cls["p_elect_g_vclass_lclass_cls"]
    )

    keep_cols = lcols + vcols + list(ocols.values())
    out = cls.loc[:, keep_cols]
    out = out.set_index(lcols + vcols).sort_index()
    return out


def filter_dwells_post_prob(dw: DwellSet, pcols: dict) -> DwellSet:
    """Filter dwells down to only include the ones we want to sample.

    That is, we want the dwells which occur within:
        - The location target pool OR the location donor pool
        - The time spans expected (e.g. weekdays)
        - AND is electrified

    DO NOT drop the zero-charge dwells. They're critical to the dwell-based sampling.
    Furthermore, I'll need to consider keeping the non-critical days around to retain
    these dwells.
    """
    if not dw.is_dask:
        old_len = len(dw.data)

    # Dwells which are feasible to electrify
    vehicle_feasible = dw.data[pcols["electrified_col"]]

    # Perform filtering
    feas = dw.copy_without_data()
    feas.data = dw.data.loc[vehicle_feasible, :]
    feas.data = feas.data.drop(columns=[pcols["electrified_col"]])

    if not dw.is_dask:
        new_len = len(dw.data)
        abs_diff = old_len - new_len
        pct_diff = round(abs_diff / old_len * 100, 1)
        logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")

    return feas


def add_dwell_id(dw: DwellSet, params: dict) -> DwellSet:
    """Add a dwell id column."""
    dw.data[params["dw_col"]] = np.arange(len(dw.data))
    return dw


def get_dwells_nonzero(dw: DwellSet, pcols: dict) -> DwellSet:
    """Get dwells down to only include the ones with nonzero charging.

    We finally drop the zero-charge dwells here, because we don't need to add them to
    the load profiles EVEN IF they are sampled.
    """
    if not dw.is_dask:
        old_len = len(dw.data)

    # Dwells which have nonzero charging
    nonzero_chg = dw.data[pcols["charge_col"]] > 0

    # Perform filtering
    nz = dw.copy_without_data()  # IMPORTANT: This is a view, not an inplace filtering
    nz.data = dw.data.loc[nonzero_chg, :]

    if not dw.is_dask:
        new_len = len(nz.data)
        abs_diff = old_len - new_len
        pct_diff = round(abs_diff / old_len * 100, 1)
        logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")

    return nz


def manage_charging(dw: DwellSet, params: dict) -> pd.DataFrame:
    """Manage the charging of vehicles within each dwell to create charging events."""
    manager_cls = _MANAGER_MAP[params["charging_manager"]]
    manager = manager_cls(
        dw=dw, **params["input_cols"], prof_type=ProfileType.OBSERVATIONS
    )
    events = manager.get_events()
    pow_col = manager_cls.suffixes["power"]
    events[pow_col] = events[pow_col].round(params["round_decimals"])
    return events


def slice_events(events: pd.DataFrame, params: dict, pcols: dict) -> pd.DataFrame:
    """Slice up an events DataFrame into vehicle-windows.

    This involves setting a frequency-based slicing, and then creating new events to
    'initialize' each new slice if there was carry-over charging from the slice previous
    in time.

    """
    dur_col = pcols["duration_col"]
    slice_begin_col = params["slice_id_col"]
    slice_time_col = params["slice_time_col"]
    src_time_col = params["source_time_col"]

    if events[src_time_col].dt.tz is not None:
        raise RuntimeError("The source time column must be time zone naïve.")

    events_mod = events.reset_index()

    # Clip durations to maximum length so that we preserve characteristic shape
    if params["clip_dur_to_slice_freq"]:
        dur_spread_col = "dur_clipped"
        max_dur = pd.Timedelta(params["slice_freq"])
        events_mod[dur_spread_col] = events_mod[dur_col].clip(upper=max_dur)
    else:
        dur_spread_col = dur_col

    logger.info(
        "Spread observations to cover all local-time slices across their duration"
    )
    grp_cols = [pcols["dw_col"]]
    cum_cols = list(pcols["profile_cols"].values())
    spreader = IntervalBeginSpreader(
        time_col=src_time_col,
        dur_col=dur_spread_col,
        value_cols=cum_cols,
        group_cols=grp_cols,
        freq=params["slice_freq"],
    )
    to_spread = events_mod.dropna(subset=dur_spread_col)
    inits = spreader.spread(to_spread, return_spreaded_only=True)

    logger.info(
        "Concatenating and sorting original observations and initialization observations."
    )
    keep_cols = grp_cols + [src_time_col] + cum_cols
    events_mod = events_mod.loc[:, keep_cols]
    events_concat = pd.concat([events_mod, inits], axis=0, ignore_index=True)
    events_concat[slice_begin_col] = events_concat[src_time_col].dt.floor(
        params["slice_freq"]
    )
    # Note that `time_local` is both local and then timezone-naïve. This means that the
    # daylight savings time extra and missing hours are introduced here
    events_concat[slice_time_col] = (
        events_concat[src_time_col] - events_concat[slice_begin_col] + pd.Timestamp(0)
    )

    events_out = events_concat.drop(columns=[src_time_col, slice_begin_col])
    return events_out


def build_time_ordered_slice(
    events: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Build the time-ordered slice of differences use for sampling from."""
    events_sort = events.sort_values(
        [pcols["dw_col"], params["slice_time_col"]], ascending=[True, True]
    )

    prof_cols = pcols["profile_cols"]
    diff_cols = pcols["diff_cols"]
    if set(prof_cols.keys()) != set(diff_cols.keys()):
        raise ValueError("Profile and diff column sets must be paired in the config.")

    for k, pc in prof_cols.items():
        events_sort[diff_cols[k]] = events_sort.groupby(pcols["dw_col"])[pc].diff()
        na_diff = events_sort[diff_cols[k]].isna()
        events_sort[diff_cols[k]] = events_sort[diff_cols[k]].where(
            ~na_diff, events_sort[pc]
        )

    events_sort = events_sort.drop(columns=list(prof_cols.values()))
    events_sort = events_sort.sort_values(params["slice_time_col"])

    return events_sort


def sample_profiles_node(
    dw: DwellSet,
    events: pd.DataFrame,
    locs: pd.DataFrame,
    classes: pd.DataFrame,
    params: dict,
    pcols: dict,
    client: Client,
) -> pd.DataFrame:
    """Sample load profiles using self and class dwells."""
    sample_self = params["sample_self"]
    sample_class = params["sample_class"]
    n_boots = params["n_bootstraps"]
    if not sample_self and not sample_class and n_boots > 1:
        logger.warning(
            "No sampling requested, so defaulting the number of bootstraps to 1."
        )
        n_boots = 1

    hex_col = pcols["hex_col"]
    hex_col_compact = f"{hex_col}_compact"

    reg_col = pcols["group_cols"][0]  # TODO: Enable multi-column
    reg_col_compact = f"{reg_col}_compact"

    logger.info("Build compact identifiers and correspondence matrices")
    locs_counts = locs.reset_index(drop=False)
    locs_counts.index.name = hex_col_compact
    locs_counts = locs_counts.reset_index(drop=False)

    mrg = locs_counts.loc[:, [hex_col, hex_col_compact]]
    dwells_samp = dw.data.merge(mrg, how="left", on=hex_col)

    locs_counts[reg_col_compact] = pd.Categorical(locs_counts[reg_col]).codes

    # Build Dwells x Locations mask
    n_hexes = locs_counts[hex_col_compact].max() + 1
    Ga = build_entity_mask_array(ids=dwells_samp[hex_col_compact].values, n_ent=n_hexes)
    Ga = Ga.tocsr()

    # Build Locations x Lclasses mask
    class_arr = locs_counts[params["loc_group_col"]].values.astype(int)
    Cy = build_entity_mask_array(ids=class_arr)
    Cy = Cy.tocsr()

    # Build Events x Dwells mask
    n_dwells = dwells_samp[pcols["dw_col"]].max() + 1
    Be = build_entity_mask_array(ids=events[pcols["dw_col"]].values, n_ent=n_dwells)
    Be = Be.tocsr()

    # Build Region x Location mask
    n_regions = locs_counts[reg_col_compact].max() + 1
    Rho = build_entity_mask_array(
        ids=locs_counts[reg_col_compact].values, n_ent=n_regions
    )
    Rho = Rho.tocsr()

    ## Build the inverse propensity weights
    om = 1 / dwells_samp[params["dwell_prob_obs_elect_col"]].values[:, np.newaxis]
    # For a Dwells x Locations matrix
    Om_hex = Ga * om
    Om_hex = Om_hex.tocsc()
    Om_hex = normalize_sparse(Om_hex, axis=0)

    # For a Dwells x Clusters matrix
    Om_cls = (Ga @ Cy) * om
    Om_cls = Om_cls.tocsc()
    Om_cls = normalize_sparse(Om_cls, axis=0)

    ## Build the dwell counts, both observed and expected
    dw_arr = np.ones(shape=(om.shape[0],), dtype=np.int64)
    m_hex_obs = Ga.T @ dw_arr
    m_class_obs = Cy.T @ m_hex_obs

    exp_dwell_rate_col = params["n_dwells_expected_elect_col"]
    loc_cls = classes.groupby(params["loc_group_col"])[exp_dwell_rate_col].sum()
    loc_cls = loc_cls.sort_index()
    n_locs_per_class = Cy.sum(axis=0)
    m_class_expected = loc_cls.values / n_locs_per_class
    m_hex_expected = Cy @ m_class_expected

    # Get profiles
    tcol = params["time_col"]
    kws = {
        "m_hex_expected": m_hex_expected,
        "m_hex_obs": m_hex_obs.astype(int),
        "m_class_expected": m_class_expected,
        "m_class_obs": m_class_obs.astype(int),
        "hex_class": class_arr,
        "max_first_stage_options": params["max_first_stage_options"],
        "Om_hex": Om_hex,
        "Om_class": Om_cls,
        "events_by_dwells": Be,
        "region_by_hex": Rho,
        "events": events,
        "slice_freq": params["slice_freq"],
        "discrete_freq": params["discrete_freq"],
        "prof_cols": list(pcols["diff_cols"].values()),
        "dur_col": pcols["duration_col"],
        "region_name": reg_col_compact,
        "time_col": tcol,
        "sample_self": params["sample_self"],
        "sample_class": params["sample_class"],
    }

    logger.info("Perform bootstrap sampling")
    prof_dict = {}

    if n_boots == 1:
        prof_dict[0] = sample_profiles(**kws)
    elif n_boots > 1:

        def _sample_profiles_distrib(kws: dict, boot_id: int):
            # boot_id argument ensures that Dask does not run once and copy results
            return sample_profiles(**kws)

        future_kws = client.scatter(kws, broadcast=True)
        futures = [
            client.submit(_sample_profiles_distrib, kws=future_kws, boot_id=boot_id)
            for boot_id in range(n_boots)
        ]
        for boot_id, (future, result) in tqdm(
            enumerate(as_completed(futures, with_results=True)), total=len(futures)
        ):
            prof_dict[boot_id] = result
    else:
        raise ValueError("Number of bootstraps must be >= 1.")

    logger.info("Concatenate bootstrap results")
    boot_profs = pd.concat(prof_dict, names=[params["bootstrap_id_col"], "index"])
    del prof_dict
    gc.collect()
    boot_profs = boot_profs.droplevel("index")
    boot_profs = boot_profs.reset_index()

    # Reset names to originals
    renamer = {d: pcols["profile_cols"][k] for k, d in pcols["diff_cols"].items()}
    boot_profs = boot_profs.rename(columns=renamer)

    loc_counts_names = locs_counts.loc[:, [reg_col_compact, reg_col]].drop_duplicates()
    reg_name_restorer = loc_counts_names.set_index(reg_col_compact)[reg_col]
    boot_profs[reg_col] = boot_profs[reg_col_compact].map(reg_name_restorer)
    boot_profs = boot_profs.drop(columns=[reg_col_compact])

    return boot_profs


def compute_location_dwell_counts(
    clusts: pd.DataFrame, classes: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Compute expected and available dwell counts per location for use in sampling."""
    lclass_cols = params["loc_class_cols"]
    n_dwl_expected_class_col = params["out_columns"]["n_elect_dwells_expected"]
    mrg = classes.groupby(lclass_cols)[n_dwl_expected_class_col].sum()
    clusts = clusts.merge(mrg, how="left", on=lclass_cols)

    mrg = clusts.groupby(lclass_cols).size()
    mrg.name = "n_locs_in_class"
    clusts = clusts.merge(mrg, how="left", on=lclass_cols)
    clusts["n_dwells_expected"] = (
        clusts[n_dwl_expected_class_col] / clusts["n_locs_in_class"]
    )
    return clusts


# def filter_events(events: pd.DataFrame, params: dict, pcols: dict) -> pd.DataFrame:
#     """Filter events down to only include the ones we want to summarize."""

#     # DO NOT drop the zero-charge dwells. They're critical to the dwell-based sampling.
#     # Furthermore, I'll need to consider keeping the non-critical days around to retain
#     # these dwells.

#     # However, dropping optional stops not taken could be very important.

#     # After apply_delays, optional stops without charging should be the only ones with
#     #   zero duration.

#     old_len = len(events)

#     prof_cols = list(pcols["diff_cols"].values())
#     any_nonzero = (events.loc[:, prof_cols] != 0).any(axis=1)
#     vehicle_feasible = ~events[params["drop_events_col"]]
#     duplic_events = events.duplicated(
#         subset=params["duplic_check_cols"], keep=False
#     )  # TODO: Move this to an event filtering step
#     events = events.loc[any_nonzero & vehicle_feasible & ~duplic_events, :]
#     events = events.dropna(subset=params["drop_na_cols"])
#     events = events.drop(columns=[params["drop_events_col"]])

#     new_len = len(events)
#     abs_diff = old_len - new_len
#     pct_diff = round(abs_diff / old_len * 100, 1)
#     logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")
#     return events


# def filter_slices_location(
#     slices: pd.DataFrame, params: dict, pcols: dict
# ) -> pd.DataFrame:
#     """Filter the vehicle-time slices (e.g. to only include weekdays).

#     This process would usually occur before sampling of the vehicle-time slices, so that
#     only relevant slice types are used.
#     """
#     # Remove slices applying to untracked locations (this had been earlier for performance)
#     slices_filt = slices.dropna(subset=pcols["group_cols"])
#     return slices_filt


# def filter_slices_time(slices: pd.DataFrame, params: dict, pcols: dict) -> pd.DataFrame:
#     """Filter the vehicle-time slices (e.g. to only include weekdays).

#     This process would usually occur before sampling of the vehicle-time slices, so that
#     only relevant slice types are used.
#     """
#     FIRST_DAY_OF_WEEKEND = 5
#     slices["is_weekend"] = (
#         slices[params["slice_id_col"]].dt.weekday >= FIRST_DAY_OF_WEEKEND
#     )
#     slices_filt = slices.loc[~slices["is_weekend"]]
#     slices_filt = slices_filt.drop(columns=["is_weekend"])
#     return slices_filt


def build_eval_columns(pcols: dict, group_cols: list) -> dict:
    """Build a set of evaluation columns to use throughout the pipeline."""
    pcols.update({"group_cols": group_cols})
    return pcols


def assign_regions(
    events: pd.DataFrame, hex_regions: pd.DataFrame, pcols: dict
) -> pd.DataFrame:
    """Assign larger regions to the DwellSet based on hexagon ids."""
    orig_idx = events.index.names
    events = events.reset_index()
    grp_cols = list(set(pcols["group_cols"]).intersection(hex_regions.columns))
    reg_cols = [pcols["hex_col"]] + grp_cols + [pcols["timezone_col"]]
    mrg = hex_regions.reset_index().loc[:, reg_cols]
    events = events.merge(mrg, how="left", on=pcols["hex_col"])
    if orig_idx != [None]:
        events = events.set_index(orig_idx)
    else:
        events = events.drop(columns=["index"])
    return events


def assign_vehicle_metadata(
    events: pd.DataFrame, vehs: pd.DataFrame, pcols: dict
) -> pd.DataFrame:
    """Assign larger regions to the DwellSet based on hexagon ids."""
    orig_idx = events.index.names
    events = events.reset_index()
    grp_cols = list(set(pcols["group_cols"]).intersection(vehs.columns))
    if len(grp_cols) > 0:
        veh_cols = [pcols["veh_col"]] + grp_cols
        mrg = vehs.reset_index().loc[:, veh_cols]
        events = events.merge(mrg, how="left", on=pcols["veh_col"])
    events = events.set_index(orig_idx)
    return events


def get_load_profiles(events: pd.DataFrame, params: dict, pcols: dict) -> pd.DataFrame:
    """Convert vehicle charging events to load profiles by group."""
    logger.info("Sorting events by group and time")
    events = events.reset_index()
    if params["drop_null_groups"]:
        events = events.dropna(subset=pcols["group_cols"])
    events = events.sort_values(pcols["group_cols"] + [pcols["time_col"]])

    logger.info("Calculating load profiles by accumulating events.")
    event_grp = events.groupby(pcols["group_cols"], sort=False, observed=True)
    # Note: The vehicle and hexagon ids are rendered uninterpretable by the cumsum
    events[pcols["profile_col"]] = event_grp[params["power_col"]].cumsum()
    events[pcols["duration_col"]] = event_grp[pcols["time_col"]].transform(
        lambda ser: ser.shift(-1) - ser
    )
    profs = events.drop(columns=[params["power_col"]])
    profs = profs.set_index(pcols["group_cols"] + [pcols["time_col"]])
    return profs


def report_by_region_peaks(
    profs: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Report results by hex."""
    orig_idx = profs.index.names
    profs = profs.reset_index()

    logger.info("Finding peaks")
    max_idx = profs.groupby(pcols["group_cols"], sort=False, observed=True)[
        pcols["profile_col"]
    ].idxmax()
    peaks = profs.loc[max_idx]

    logger.info("Calculating local time attributes")
    peaks = calc_local_time(
        df=peaks,
        time_cols=pcols["time_col"],
        local_cols=pcols["time_col"] + "_local",
        tz_col=pcols["timezone_col"],
    )
    peaks = peaks.set_index(orig_idx)
    return peaks


def report_by_region_quantiles(
    profs: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Report quantile summaries by region and time grouping."""
    grped_nonzero = discretize_sparse_profiles(
        profs=profs,
        time_col=pcols["time_col"],
        dur_col=pcols["duration_col"],
        prof_col=pcols["profile_col"],
        tz_col=pcols["timezone_col"],
        group_cols=pcols["group_cols"],
        freq=params["freq"],
    )

    tgpars = params["time_grouper"]
    grouper = HourOfWeekdayGrouper(
        time_col=pcols["time_col"],
        tz_col=pcols["timezone_col"],
        start_time=pd.Timestamp(tgpars["start_time"]),
        end_time=pd.Timestamp(tgpars["end_time"]),
        possible_tzs=tgpars["possible_tzs"],
    )
    grped_nonzero = grouper.add_group_classes(grped_nonzero)
    group_counts_tz = grouper.get_possible_obs_counts()
    group_counts_tz = group_counts_tz.reset_index()
    grp_merge_cols = [pcols["timezone_col"]] + grouper.time_group_cols
    grped_nonzero = grped_nonzero.merge(group_counts_tz, how="left", on=grp_merge_cols)

    logger.info("Calculate quantiles")
    summ_cols = pcols["group_cols"] + grouper.time_group_cols
    summer = NonzeroGroupedSummarizer(
        group_cols=summ_cols,
        quantiles=np.array(params["quantiles"]),
    )
    quantiles = summer.summarize(
        events=grped_nonzero,
        value_cols=pcols["profile_col"],
        possible_count_col="possible_count",
    )
    return quantiles


def add_region_geoms(
    results: pd.DataFrame,
    hex_regions: pd.DataFrame,
    params: dict,
) -> gpd.GeoDataFrame:
    """Add region geometries to the reporting by region."""
    reg_polys = cells_to_region_polygons(
        corresp=hex_regions.reset_index(),
        hex_col=params["hex_col"],
        region_col=params["region_col"],
    )
    results = results.merge(reg_polys, on=params["region_col"])
    res_geos = gpd.GeoDataFrame(results, geometry="geometry")

    changed_cols = []
    for col in res_geos.columns:
        if pd.api.types.is_timedelta64_dtype(res_geos[col]):
            res_geos[f"{col}_hrs"] = total_hours(res_geos[col])
            changed_cols.append(col)

    res_geos = res_geos.drop(columns=changed_cols)

    return res_geos


def build_slice_frame(vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Build the frame of all potential slices.

    We assume that the only valid dates for each vehicle are those beginning from its
    first observation date and extending to its last observation date. If the vehicle
    was dormant during extended periods at the beginning or end of the data collection
    period, then this would tend to overestimate the number of active days. However,
    we believe that this bias is better than the bias of assuming that all vehicles
    are observed for the same period of time, which would lead to a large underestimate
    of the number of active days.
    """
    pcols = params["columns_input"]
    vehs = calc_time_zones_from_hexes(
        df=vehs,
        hex_col=pcols["first_hex"],
        tz_col="tz_first",
    )
    vehs = calc_time_zones_from_hexes(
        df=vehs,
        hex_col=pcols["last_hex"],
        tz_col="tz_last",
    )
    vehs = calc_local_time(
        df=vehs,
        time_cols=pcols["first_time"],
        local_cols="obs_time_first_local",
        tz_col="tz_first",
    )
    vehs = calc_local_time(
        df=vehs,
        time_cols=pcols["last_time"],
        local_cols="obs_time_last_local",
        tz_col="tz_last",
    )
    vehs["range_start"] = vehs["obs_time_first_local"].dt.floor(params["freq"])
    vehs["range_end"] = vehs["obs_time_last_local"].dt.ceil(params["freq"])

    all_vehs = vehs.index.to_series().values
    wide_range = pd.date_range(
        start=vehs["range_start"].min(),
        end=vehs["range_end"].max(),
        freq=params["freq"],
    ).to_series()
    fcols = params["frame_cols"]
    frame = pd.MultiIndex.from_product([all_vehs, wide_range])
    frame.names = [fcols["vehicle"], fcols["time"]]
    frame = frame.to_frame().reset_index(drop=True)
    mrg = vehs.loc[:, ["range_start", "range_end"]]
    frame = frame.merge(mrg, how="left", on=fcols["vehicle"])
    too_early = frame[fcols["time"]] < frame["range_start"]
    too_late = frame[fcols["time"]] >= frame["range_end"]
    drop_idx = frame.loc[too_early | too_late].index
    frame = frame.drop(index=drop_idx, columns=["range_start", "range_end"])
    return frame


def localize_time_from_hexes(
    df: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Localize a given time column using the location provided by hexagons."""
    if params["get_tz"]:
        df = calc_time_zones_from_hexes(
            df=df,
            hex_col=pcols["hex_col"],
            tz_col=pcols["timezone_col"],
        )
    df = calc_local_time(
        df=df,
        time_cols=params["time_cols_source"],
        local_cols=params["time_cols_local"],
        tz_col=pcols["timezone_col"],
    )
    if params["sort_result"]:
        df = df.sort_values(params["sort_cols"])
    return df


class SliceWeightSampler:
    """Manage the sampling of time-sorted slices by applying a column of weights."""

    def __init__(
        self, strat_cols: list[str], target_count_col: str, verbose: bool = False
    ):
        self.strat_cols = strat_cols
        self.target_count_col = target_count_col
        self.verbose = verbose
        self.avail = None
        self.source_idx_name = None

    def prepare(
        self, sources: pd.DataFrame, frame: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        """Prepare to sample by getting the target counts and available values."""
        if not pd.api.types.is_integer_dtype(sources.index):
            raise RuntimeError(
                "The index of the sources DataFrame must have an integer dtype."
            )
        self.source_idx_name = sources.index.name

        trgts = targets.groupby(self.strat_cols, observed=True)[
            self.target_count_col
        ].sum()
        trgts.name = "n_samples"
        trgts = trgts.astype(int)
        trgts = trgts.reset_index()

        srcs = sources.groupby(self.strat_cols, observed=True).apply(
            lambda grp: grp.index.unique().to_series().values,
            include_groups=False,
        )
        srcs.name = "choice_indices"
        srcs = srcs.reset_index()
        srcs["n_nonempty_slices"] = srcs["choice_indices"].transform(lambda x: x.size)

        srcs_w_zeros = frame.groupby(self.strat_cols, observed=True).size()
        srcs_w_zeros.name = "n_possible_slices"
        srcs_w_zeros = srcs_w_zeros.reset_index()

        avail = trgts.merge(srcs_w_zeros, how="left", on=self.strat_cols)
        avail = avail.merge(srcs, how="left", on=self.strat_cols)
        avail = avail.set_index(self.strat_cols)
        avail["frac_nonempty"] = avail["n_nonempty_slices"] / avail["n_possible_slices"]

        lost_strata = list(avail.loc[avail["n_nonempty_slices"].isna()].index.values)
        if len(lost_strata) > 0 and self.verbose:
            for s in lost_strata:
                logger.warning(
                    f"No source observations available from the stratum: {s}"
                )
        avail = avail.drop(index=lost_strata)
        self.avail = avail

    def sample(self, seed: int, weight_col_name: str) -> pd.DataFrame:
        """Produce a dataframe of sample weights, which can be merged onto slices."""
        rng = np.random.default_rng(seed=seed)
        n_nonempty_samples = rng.binomial(
            n=self.avail["n_samples"],
            p=self.avail["frac_nonempty"],
        )
        sample_idx = np.zeros(n_nonempty_samples.sum(), dtype=int)
        i = 0
        for strata_idx in range(n_nonempty_samples.shape[0]):
            n_samps = n_nonempty_samples[strata_idx]
            choices = self.avail["choice_indices"].iloc[strata_idx]
            cur_samp_idx = rng.choice(choices, size=n_samps, replace=True)
            sample_idx[i : i + n_samps] = cur_samp_idx
            i += n_samps

        sample_idx, sample_counts = np.unique(sample_idx, return_counts=True)
        sample_weights = pd.DataFrame.from_dict(
            {
                self.source_idx_name: sample_idx,
                weight_col_name: sample_counts,
            }
        )
        sample_weights = sample_weights.set_index(self.source_idx_name)
        return sample_weights


def process_sample(
    winds: pd.DataFrame,
    smpler: SliceWeightSampler,
    discrete_freq: str,
    pcols: dict,
    params_slice: dict,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process a single bootstrap sample iteration."""
    if seed is not None:
        sample_weights = smpler.sample(seed=seed, weight_col_name="samp_wgt")
        samps = winds.merge(
            sample_weights, how="left", left_index=True, right_index=True
        )
        samps["samp_wgt"] = samps["samp_wgt"].fillna(0.0)
    else:
        samps = winds.copy()
        samps["samp_wgt"] = 1.0

    # Calculate profiles and energies
    for val_name, diff_col in pcols["diff_cols"].items():
        samps[f"weighted_{val_name}"] = samps[diff_col] * samps["samp_wgt"]

    samps_grp = samps.groupby(pcols["group_cols"], sort=False, observed=True)

    for val_name, prof_col in pcols["profile_cols"].items():
        samps[prof_col] = samps_grp[f"weighted_{val_name}"].cumsum()

    del samps_grp
    gc.collect()

    samps["energy_kwh"] = samps[pcols["profile_cols"]["power"]] * samps["dur_hrs"]
    samps_grp = samps.groupby(pcols["group_cols"], sort=False, observed=True)
    energies = samps_grp["energy_kwh"].sum()
    del samps_grp
    gc.collect()

    with SuppressLogs():
        discs = discretize_sparse_profiles(
            profs=samps,
            time_col=params_slice["slice_time_col"],
            dur_col=pcols["duration_col"],
            prof_cols=list(pcols["profile_cols"].values()),
            group_cols=pcols["group_cols"],
            freq=discrete_freq,
        )

    # Clean up to reduce memory usage
    del samps
    gc.collect()
    return (discs, energies)


def sample_vehicle_windows(
    windows: pd.DataFrame,
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    params_slice: dict,
    params_sample: dict,
    pcols: dict,
    client: Client,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample vehicle windows in different categories up to desired numbers."""
    if set(pcols["profile_cols"].keys()) != set(pcols["diff_cols"].keys()):
        raise ValueError(
            "The profile_cols and diff_cols dictionary keys must be the same set."
        )

    slice_time_col = params_slice["slice_time_col"]

    logger.info("Sort and set index for vehicle windows")
    winds = windows.reset_index()
    sort_cols = pcols["group_cols"] + [slice_time_col]
    sort_ord = [True] * len(sort_cols)
    winds = winds.sort_values(sort_cols, ascending=sort_ord)

    winds = winds.set_index([pcols["veh_col"], params_slice["slice_id_col"]])
    inter = IndexIntegerizer(int_col="wind_id")
    winds = inter.integerize(winds)

    logger.info("Pre-calculate duration of the profile events")
    wind_grp = winds.groupby(pcols["group_cols"], sort=False, observed=True)
    winds[pcols["duration_col"]] = wind_grp[slice_time_col].transform(
        lambda ser: ser.shift(-1) - ser
    )
    winds["dur_hrs"] = total_hours(winds[pcols["duration_col"]])

    logger.info("Prepare sampler")
    smpler = SliceWeightSampler(
        strat_cols=params_sample["stratify_cols"],
        target_count_col=params_sample["target_count_col"],
        verbose=False,
    )
    smpler.prepare(sources=winds, frame=frame, targets=targets)

    keep_cols = (
        pcols["group_cols"]
        + [slice_time_col]
        + list(pcols["diff_cols"].values())
        + [pcols["duration_col"], "dur_hrs"]
    )
    drop_cols = set(winds.columns).difference(keep_cols)
    winds_mini = winds.drop(columns=drop_cols)
    del winds
    gc.collect()

    # Build seeds
    if not params_sample["skip_resampling"]:
        rng = np.random.default_rng(seed=params_sample["seed_for_seeds"])
        seeds = rng.integers(0, 1_000_000, size=params_sample["n_bootstraps"])
    else:
        seeds = [None]

    # Create delayed tasks
    logger.info("Scatter datasets to distributed memory")
    future_winds = client.scatter(winds_mini, broadcast=True)
    future_smpler = client.scatter(smpler, broadcast=True)

    logger.info("Compute samples")
    futures = [
        client.submit(
            process_sample,
            winds=future_winds,
            smpler=future_smpler,
            discrete_freq=params_sample["discrete_freq"],
            pcols=pcols,
            params_slice=params_slice,
            seed=seed,
        )
        for seed in seeds
    ]

    # Process results
    prof_dict = {}
    energy_dict = {}
    for i, (future, result) in tqdm(
        enumerate(as_completed(futures, with_results=True)), total=len(futures)
    ):
        prof, energy = result
        prof_dict[i] = prof
        energy_dict[i] = energy

    del futures, future_winds, future_smpler, future

    logger.info("Collect and concatenate samples")
    # Note: With parallel sampling, the bootstrap id is no longer deterministic
    boot_profs = pd.concat(prof_dict, names=[params_slice["bootstrap_id_col"], "index"])
    boot_profs = boot_profs.droplevel("index")
    boot_profs = boot_profs.reset_index()

    boot_energies = pd.concat(energy_dict, names=[params_slice["bootstrap_id_col"]])
    boot_energies.name = "energy_delivered_kwh"
    boot_energies = boot_energies.reset_index()

    return (boot_profs, boot_energies)


def summarize_vehicle_window_quantiles(
    profs: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Summarize vehicle windows by grouping into times and then quantiling."""
    tcol = params["time_col"]
    grouper = AdaptiveTimeGrouper(
        time_col=tcol,
        tz_col=pcols["timezone_col"],
        start_time=pd.Timestamp(0),
        end_time=pd.Timestamp(0) + pd.Timedelta(params["slice_freq"]),
        possible_tzs=["no_time_zone"],
        freq=params["discrete_freq"],
    )
    profs = grouper.add_group_classes(profs)
    grp_cnts_tz = grouper.get_possible_obs_counts()
    n_bootstraps = profs[params["bootstrap_id_col"]].nunique()
    grp_cnts_tz = grp_cnts_tz * n_bootstraps
    grp_cnts_tz = grp_cnts_tz.reset_index()
    grp_cnts_tz[tcol] = grp_cnts_tz[tcol].dt.tz_localize(None)
    grp_merge_cols = grouper.time_group_cols
    profs = profs.merge(grp_cnts_tz, how="left", on=grp_merge_cols)

    logger.info("Calculate quantiles")
    summ_cols = pcols["group_cols"] + grouper.time_group_cols
    summer = NonzeroGroupedSummarizer(
        group_cols=summ_cols,
        quantiles=np.array(params["quantiles"]),
    )
    quantiles = summer.summarize(
        events=profs,
        value_cols=list(pcols["profile_cols"].values()),
        possible_count_col="possible_count",
    )
    return quantiles
