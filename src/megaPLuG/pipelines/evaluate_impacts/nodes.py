"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import gc
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed
from tqdm.auto import tqdm

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.models.group_times import AdaptiveTimeGrouper, HourOfWeekdayGrouper
from megaPLuG.models.summarize import EventExpander, NonzeroGroupedSummarizer
from megaPLuG.utils.data import IndexIntegerizer, filter_by_vals_in_cols
from megaPLuG.utils.h3 import cells_to_region_polygons
from megaPLuG.utils.logging import SuppressLogs
from megaPLuG.utils.time import (
    HOURS_PER_WEEK,
    calc_local_time,
    calc_time_zones_from_hexes,
    total_hours,
)

logger = logging.getLogger(__name__)


# ruff: noqa: PLR0915
def summarize_vehicles(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Summarize the results for each vehicle."""
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

    n_deaths = dw.data.groupby(dw.veh, sort=False).agg(
        n_deaths_all=pd.NamedAgg("is_death", "sum"),
        n_deaths_circle=pd.NamedAgg("is_death_circle", "sum"),
        dead_hrs=pd.NamedAgg("dead_hrs", "sum"),
    )
    n_deaths["n_deaths_addressable"] = (
        n_deaths["n_deaths_all"] - n_deaths["n_deaths_circle"]
    )
    vehs = vehs.merge(n_deaths, how="inner", on=dw.veh)
    hrs_obs = total_hours(vehs[params["obs_duration_col"]])
    weeks_obs = hrs_obs / HOURS_PER_WEEK
    vehs["n_deaths_all_per_week"] = vehs["n_deaths_all"] / weeks_obs
    vehs["n_deaths_circle_per_week"] = vehs["n_deaths_circle"] / weeks_obs
    vehs["n_deaths_addressable_per_week"] = vehs["n_deaths_addressable"] / weeks_obs
    vehs["frac_dead_hrs"] = vehs["dead_hrs"] / hrs_obs

    logger.info("Deaths per week per vehicle (all, including circle trips):")
    logger.info(vehs["n_deaths_all_per_week"].describe())

    logger.info("Deaths per week per vehicle (addressable):")
    logger.info(vehs["n_deaths_addressable_per_week"].describe())

    logger.info("Fraction of observed hours spent in dead state per vehicle:")
    logger.info(vehs["frac_dead_hrs"].describe())

    # Delay as fraction of shift duration for each vehicle
    dw.data["shift_id"] = dw.data.groupby(dw.veh)[params["shift_refresh_col"]].cumsum()
    dw.data["shift_id"] = dw.data.groupby(dw.veh)["shift_id"].shift(1, fill_value=0)
    dw.data["delay_inc_hrs_shift"] = dw.data.groupby(dw.veh)[
        params["delay_inc_hrs_col"]
    ].shift(1, fill_value=0)

    delays = dw.data.groupby([dw.veh, "shift_id"]).agg(
        max_delay_hrs=pd.NamedAgg(params["delay_hrs_col"], "max"),
        prev_delay_hrs_inc=pd.NamedAgg("delay_inc_hrs_shift", "first"),
        init_delay_hrs=pd.NamedAgg(params["delay_hrs_col"], "first"),
        final_delay_hrs=pd.NamedAgg(params["delay_hrs_col"], "last"),
        trip_dur_sum=pd.NamedAgg(dw.trip_dur, "sum"),
        dwell_dur_sum=pd.NamedAgg(params["dwell_dur_col"], "sum"),
        dwell_dur_last=pd.NamedAgg(params["dwell_dur_col"], "last"),
    )

    scols = params["summary_cols"]
    delays = delays.rename(columns={"max_delay_hrs": scols["max_delay"]})
    # The delay incurred by this shift is: the cumulative delay level when the vehicle
    # arrives at the final depot MINUS the cumulative delay level when the vehicle arrives
    # at its first (non-refresh) stop in the shift PLUS the increase in delay incurred
    # at the refresh stop directly before this shift (in preparation for this shift)
    delays[scols["shift_delay"]] = (
        delays["final_delay_hrs"]
        - delays["init_delay_hrs"]
        + delays["prev_delay_hrs_inc"]
    )
    # Vehicle death can sometimes cause cumulative delay to drop, so we ignore those
    # shifts, setting their delay to zero
    delays[scols["shift_delay"]] = delays[scols["shift_delay"]].clip(lower=0.0)

    delays[scols["shift_dur"]] = (
        delays["trip_dur_sum"] + delays["dwell_dur_sum"] - delays["dwell_dur_last"]
    )
    delays[scols["shift_dur_delayed"]] = (
        delays[scols["shift_dur"]] + delays[scols["shift_delay"]]
    )
    delays[scols["delay_frac"]] = (
        delays[scols["shift_delay"]] / delays[scols["shift_dur"]]
    )

    delays_grp = delays.groupby(dw.veh)
    veh_delays_rel = delays_grp[list(scols.values())].quantile(params["quantiles"])
    veh_delays_rel = veh_delays_rel.unstack(level=1)

    def build_col_name(val: str, pct: float) -> str:
        return f"{val}_{int(pct * 100)}"

    flat_cols = [build_col_name(name, pct) for name, pct in veh_delays_rel.columns]
    veh_delays_rel.columns = flat_cols
    vehs = vehs.merge(veh_delays_rel, how="left", on=dw.veh)

    thresh_qtl = params["delay_frac_thresh_quantile"]
    report_pctl = int(thresh_qtl * 100)
    logger.info(
        f"Delay as fraction of vehicle shift length per vehicle [{report_pctl}th percentile]:"
    )
    report_col = build_col_name(scols["delay_frac"], thresh_qtl)
    logger.info(vehs[report_col].describe())

    thresh_qtl = params["max_delay_thresh_quantile"]
    report_pctl = int(thresh_qtl * 100)
    logger.info(
        f"Maximum absolute delay accumulated per vehicle [{report_pctl}th percentile]:"
    )
    report_col = build_col_name(scols["max_delay"], thresh_qtl)
    logger.info(vehs[report_col].describe())

    # Get boolean columns for which vehicles are included in load profiles
    thrs = params["thresholds"]
    vehs["dies_too_freq"] = (
        vehs["n_deaths_all_per_week"] > thrs["n_deaths_per_week_max"]
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

    logger.info("Vehicles to be dropped [%] based on reason:")
    reasons = [
        "dies_too_freq",
        "delays_too_long_rel",
        "delays_too_long_abs",
        params["drop_events_col"],
    ]
    logger.info(np.round(vehs.loc[:, reasons].sum(axis=0) / len(vehs) * 100, 2))
    return vehs


def filter_events(events: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Filter events down to only include the ones we want to summarize."""
    old_len = len(events)

    nonzero_power = events[params["power_col"]] != 0
    vehicle_feasible = ~events[params["drop_events_col"]]
    events = events.loc[nonzero_power & vehicle_feasible, :]
    events = events.dropna(subset=params["drop_na_cols"])
    events = events.drop(columns=[params["drop_events_col"]])

    new_len = len(events)
    abs_diff = old_len - new_len
    pct_diff = round(abs_diff / old_len * 100, 1)
    logger.info(f"Rows dropped: {abs_diff}, {pct_diff}%")
    return events


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


def discretize_sparse_profiles(
    profs: pd.DataFrame,
    time_col: str,
    dur_col: str,
    prof_col: str,
    group_cols: list[str],
    tz_col: str | None = None,
    freq: str = "1h",
) -> pd.DataFrame:
    """Discretize profiles by region and time grouping."""
    logger.info("Remove all observations with unknown duration or zero power.")
    # First drop the observations with no duration or zero power
    is_na_dur = profs[dur_col].isna()
    is_zero_prof = profs[prof_col] == 0
    nonzero = profs.loc[~is_na_dur & ~is_zero_prof, :]

    logger.info("Expand events to cover all groups across their duration")
    if tz_col is not None:
        grp_cols = group_cols + [tz_col]
    else:
        grp_cols = group_cols
    expander = EventExpander(
        time_col=time_col,
        dur_col=dur_col,
        value_col=prof_col,
        group_cols=grp_cols,
        freq=freq,
    )
    nonzero_exp = expander.expand_events(nonzero)

    logger.info("Group events")
    grouper = grp_cols + [pd.Grouper(key=time_col, freq=freq)]
    gpby = nonzero_exp.groupby(grouper, sort=False, observed=True)
    grped_nonzero = gpby[prof_col].max()
    grped_nonzero = grped_nonzero.reset_index()
    return grped_nonzero


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
        value_col=pcols["profile_col"],
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


def slice_vehicle_windows(
    events: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Slice up an events DataFrame into vehicle-windows.

    This involves setting a frequency-based slicing, and then creating new events to
    'initialize' each new slice if there was carry-over charging from the slice previous
    in time.

    """
    PROF_COL = "veh_prof"
    DUR_COL = "duration"
    slice_begin_col = params["slice_id_col"]
    slice_time_col = params["slice_time_col"]
    src_time_col = params["source_time_col"]

    orig_idx = events.index.names
    events_mod = events.reset_index()

    if events[src_time_col].dt.tz is not None:
        raise RuntimeError("The source time column must be time zone naïve.")

    # Descending sorting on power_col ensures that zero-time events will not create negative charging.
    logger.info("Sort by vehicle and time")
    sort_cols = [pcols["veh_col"], src_time_col, params["power_col"]]
    events_mod = events_mod.sort_values(sort_cols, ascending=[True, True, False])

    logger.info("Build profile and duration columns for internal use (will be dropped)")
    events_grp = events_mod.groupby(pcols["veh_col"], sort=False, observed=True)
    events_mod[PROF_COL] = events_grp[params["power_col"]].cumsum()
    events_mod[DUR_COL] = events_grp[src_time_col].transform(
        lambda ser: ser.shift(-1) - ser
    )

    nonzero = events_mod.dropna(subset=[DUR_COL])
    nonzero = nonzero.reset_index()
    drop_idx = nonzero.loc[nonzero[PROF_COL] == 0].index
    nonzero = nonzero.drop(index=drop_idx)

    logger.info("Expand events to cover all slices across their duration")
    grp_cols = [
        pcols["veh_col"],
        pcols["hex_col"],
    ]  # Including hex col so that it stays integer
    expander = EventExpander(
        time_col=src_time_col,
        dur_col=DUR_COL,
        value_col=PROF_COL,
        group_cols=grp_cols,
        freq=params["slice_freq"],
    )
    inits = expander.expand_events(nonzero, return_expansions_only=True)
    inits = inits.rename(columns={PROF_COL: params["power_col"]})
    inits["is_init"] = True

    logger.info(
        "Concatenating and sorting original observations and initialization 'observations'."
    )
    events_concat = events_mod.drop(columns=[PROF_COL, DUR_COL])
    events_concat["is_init"] = False
    events_windows = pd.concat([events_concat, inits], axis=0, ignore_index=True)
    events_windows[slice_begin_col] = events_windows[src_time_col].dt.floor(
        params["slice_freq"]
    )
    # Note that `time_local` is both local and then timezone-naïve. This means that the
    # daylight savings time extra and missing hours are introduced here
    events_windows[slice_time_col] = (
        events_windows[src_time_col] - events_windows[slice_begin_col] + pd.Timestamp(0)
    )
    events_windows = events_windows.sort_values(
        sort_cols, ascending=[True, True, False]
    )
    events_windows = events_windows.ffill()

    if orig_idx != [None]:
        events_windows = events_windows.set_index(orig_idx)
    return events_windows


def filter_slices_location(
    slices: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Filter the vehicle-time slices (e.g. to only include weekdays).

    This process would usually occur before sampling of the vehicle-time slices, so that
    only relevant slice types are used.
    """
    # Remove slices applying to untracked locations (this had been earlier for performance)
    slices_filt = slices.dropna(subset=pcols["group_cols"])
    return slices_filt


def filter_slices_time(slices: pd.DataFrame, params: dict, pcols: dict) -> pd.DataFrame:
    """Filter the vehicle-time slices (e.g. to only include weekdays).

    This process would usually occur before sampling of the vehicle-time slices, so that
    only relevant slice types are used.
    """
    FIRST_DAY_OF_WEEKEND = 5
    slices["is_weekend"] = (
        slices[params["slice_id_col"]].dt.weekday >= FIRST_DAY_OF_WEEKEND
    )
    slices_filt = slices.loc[~slices["is_weekend"]]
    slices_filt = slices_filt.drop(columns=["is_weekend"])
    return slices_filt


def build_sampling_totals(scaler: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Build sampling totals for this particular scenario run."""
    tots = filter_by_vals_in_cols(scaler, params["filter_totals"])
    tot_col = params["totals_column"]
    tots.loc[:, tot_col] = tots[tot_col].astype(int)
    return tots


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
    samps["weighted_power"] = samps[params_slice["power_col"]] * samps["samp_wgt"]
    samps_grp = samps.groupby(pcols["group_cols"], sort=False, observed=True)
    samps[pcols["profile_col"]] = samps_grp["weighted_power"].cumsum()
    del samps_grp
    gc.collect()

    samps["energy_kwh"] = samps[pcols["profile_col"]] * samps["dur_hrs"]
    samps_grp = samps.groupby(pcols["group_cols"], sort=False, observed=True)
    energies = samps_grp["energy_kwh"].sum()
    del samps_grp
    gc.collect()

    with SuppressLogs():
        discs = discretize_sparse_profiles(
            profs=samps,
            time_col=params_slice["slice_time_col"],
            dur_col=pcols["duration_col"],
            prof_col=pcols["profile_col"],
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
    slice_time_col = params_slice["slice_time_col"]

    logger.info("Sort and set index for vehicle windows")
    # Sorting on impulse value is essential to keep all power levels positive, in the
    # case of zero-time events
    winds = windows.reset_index()
    sort_cols = pcols["group_cols"] + [slice_time_col, params_slice["power_col"]]
    sort_ord = [True] * (len(sort_cols) - 1) + [False]
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

    keep_cols = pcols["group_cols"] + [
        slice_time_col,
        params_slice["power_col"],
        pcols["duration_col"],
        "dur_hrs",
    ]
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
    profs: pd.DataFrame, params_slice: dict, params_summ: dict, pcols: dict
) -> pd.DataFrame:
    """Summarize vehicle windows by grouping into times and then quantiling."""
    tcol = params_slice["slice_time_col"]
    grouper = AdaptiveTimeGrouper(
        time_col=tcol,
        tz_col=pcols["timezone_col"],
        start_time=pd.Timestamp(0),
        end_time=pd.Timestamp(0) + pd.Timedelta(params_slice["slice_freq"]),
        possible_tzs=["no_time_zone"],
        freq=params_summ["discrete_freq"],
    )
    profs = grouper.add_group_classes(profs)
    grp_cnts_tz = grouper.get_possible_obs_counts()
    n_bootstraps = profs[params_slice["bootstrap_id_col"]].nunique()
    grp_cnts_tz = grp_cnts_tz * n_bootstraps
    grp_cnts_tz = grp_cnts_tz.reset_index()
    grp_cnts_tz[tcol] = grp_cnts_tz[tcol].dt.tz_localize(None)
    grp_merge_cols = grouper.time_group_cols
    profs = profs.merge(grp_cnts_tz, how="left", on=grp_merge_cols)

    logger.info("Calculate quantiles")
    summ_cols = pcols["group_cols"] + grouper.time_group_cols
    summer = NonzeroGroupedSummarizer(
        group_cols=summ_cols,
        quantiles=np.array(params_summ["quantiles"]),
    )
    quantiles = summer.summarize(
        events=profs,
        value_col=pcols["profile_col"],
        possible_count_col="possible_count",
    )
    return quantiles
