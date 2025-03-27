"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.models.group_times import HourOfWeekdayGrouper, LocalHourOfDayGrouper
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


def summarize_vehicles(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Summarize the results for each vehicle."""
    # Number of deaths by vehicle
    dw.data["is_death"] = dw.data[params["dead_energy_col"]] < 0
    n_deaths = dw.data.groupby(dw.veh, sort=False)["is_death"].sum()
    n_deaths.name = "n_deaths"
    vehs = vehs.merge(n_deaths, how="inner", on=dw.veh)
    weeks_obs = total_hours(vehs[params["obs_duration_col"]]) / HOURS_PER_WEEK
    vehs["n_deaths_per_week"] = vehs["n_deaths"] / weeks_obs

    logger.info("Deaths per week per vehicle:")
    logger.info(vehs["n_deaths_per_week"].describe())

    # Delay as fraction of shift duration for each vehicle
    dw.data["shift_id"] = dw.data.groupby(dw.veh)[params["shift_refresh_col"]].cumsum()
    dw.data["shift_id"] = dw.data.groupby(dw.veh)["shift_id"].shift(1, fill_value=0)
    dw.data["trip_start_time"] = dw.data.groupby(dw.veh)[dw.end].shift(1)

    delays = dw.data.groupby([dw.veh, "shift_id"]).agg(
        max_delay_hrs=pd.NamedAgg("dwell_init_delay_hrs", "max"),
        shift_start=pd.NamedAgg("trip_start_time", "first"),
        shift_end=pd.NamedAgg(dw.end, "last"),
    )
    delays = delays.dropna(subset=["shift_start"])

    dfcol = params["delay_frac_col"]
    delays["shift_dur_hrs"] = total_hours(delays["shift_end"] - delays["shift_start"])
    delays[dfcol] = delays["max_delay_hrs"] / delays["shift_dur_hrs"]
    veh_delays = delays.groupby(dw.veh)[dfcol].quantile(params["delay_frac_quantiles"])
    veh_delays = veh_delays.unstack(level=1)
    delay_pct_cols = [
        f"{params['delay_pct_prefix']}_{int(col * 100)}" for col in veh_delays.columns
    ]
    veh_delays.columns = delay_pct_cols
    vehs = vehs.merge(veh_delays, how="left", on=dw.veh)

    for i, col in enumerate(delay_pct_cols):
        pctl = int(params["delay_frac_quantiles"][i] * 100)
        logger.info(
            f"Delay as fraction of vehicle shift length per vehicle [{pctl}th percentile]:"
        )
        logger.info(veh_delays[col].describe())
    return vehs


def filter_events(events: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Filter events down to only include the ones we want to summarize."""
    events = events.loc[events[params["power_col"]] != 0]
    events = events.dropna(subset=params["drop_na_cols"])
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
    tz_col: str,
    group_cols: list[str],
    freq: str = "1h",
) -> pd.DataFrame:
    """Discretize profiles by region and time grouping."""
    logger.info("Remove all observations with unknown duration or zero power.")
    # First drop the observations with no duration or zero power
    nonzero = profs.dropna(subset=[dur_col])
    nonzero = nonzero.reset_index()
    drop_idx = nonzero.loc[nonzero[prof_col] == 0].index
    nonzero = nonzero.drop(index=drop_idx)

    logger.info("Expand events to cover all groups across their duration")
    grp_tz_cols = group_cols + [tz_col]
    expander = EventExpander(
        time_col=time_col,
        dur_col=dur_col,
        value_col=prof_col,
        group_cols=grp_tz_cols,
        freq=freq,
    )
    nonzero_exp = expander.expand_events(nonzero)

    logger.info("Group events")
    grouper = grp_tz_cols + [pd.Grouper(key=time_col, freq=freq)]
    grped_nonzero = nonzero_exp.groupby(grouper)[prof_col].max()
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
        self.avail["n_nonempty_samples"] = rng.binomial(
            n=self.avail["n_samples"],
            p=self.avail["frac_nonempty"],
        )
        sample_idx = np.zeros(self.avail["n_nonempty_samples"].sum(), dtype=int)
        i = 0
        for strata, row in self.avail.iterrows():
            n_samps = row["n_nonempty_samples"]
            cur_samp_idx = rng.choice(row["choice_indices"], size=n_samps, replace=True)
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


def sample_vehicle_windows(
    windows: pd.DataFrame,
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    params_slice: dict,
    params_sample: dict,
    pcols: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample vehicle windows in different categories up to desired numbers."""
    slice_time_col = params_slice["slice_time_col"]

    # Sorting on impulse value is essential to keep all power levels positive, in the case of zero-time events
    winds = windows.reset_index()
    sort_cols = pcols["group_cols"] + [slice_time_col, params_slice["power_col"]]
    winds = winds.sort_values(sort_cols, ascending=[True, True, False])

    winds = winds.set_index([pcols["veh_col"], params_slice["slice_id_col"]])
    inter = IndexIntegerizer(int_col="wind_id")
    winds = inter.integerize(winds)

    # Pre-calculate duration of the profile events
    wind_grp = winds.groupby(pcols["group_cols"], sort=False, observed=True)
    winds[pcols["duration_col"]] = wind_grp[slice_time_col].transform(
        lambda ser: ser.shift(-1) - ser
    )
    winds["dur_hrs"] = total_hours(winds[pcols["duration_col"]])

    # Prepare sampler
    smpler = SliceWeightSampler(
        strat_cols=params_sample["stratify_cols"],
        target_count_col=params_sample["target_count_col"],
        verbose=False,
    )
    smpler.prepare(sources=winds, frame=frame, targets=targets)

    # Build seeds
    if not params_sample["skip_resampling"]:
        n_bootstraps = params_sample["n_bootstraps"]
        rng = np.random.default_rng(seed=params_sample["seed_for_seeds"])
        seeds = rng.integers(0, 1_000_000, size=n_bootstraps)
    else:
        n_bootstraps = 1
        samps = winds
        samps["samp_wgt"] = 1.0

    prof_dict = {}
    energy_dict = {}
    for i in tqdm(range(n_bootstraps)):
        if not params_sample["skip_resampling"]:
            sample_weights = smpler.sample(seed=seeds[i], weight_col_name="samp_wgt")
            samps = winds.merge(
                sample_weights, how="left", left_index=True, right_index=True
            )
            samps["samp_wgt"] = samps["samp_wgt"].fillna(0.0)

        samps["weighted_power"] = samps[params_slice["power_col"]] * samps["samp_wgt"]
        samps_grp = samps.groupby(pcols["group_cols"], sort=False, observed=True)
        samps[pcols["profile_col"]] = samps_grp["weighted_power"].cumsum()
        samps["energy_kwh"] = samps[pcols["profile_col"]] * samps["dur_hrs"]
        energies = samps.groupby(pcols["group_cols"])["energy_kwh"].sum()
        with SuppressLogs():
            discs = discretize_sparse_profiles(
                profs=samps,
                time_col=slice_time_col,
                dur_col=pcols["duration_col"],
                prof_col=pcols["profile_col"],
                tz_col=pcols["timezone_col"],
                group_cols=pcols["group_cols"],
                freq=params_sample["discrete_freq"],
            )
        prof_dict[i] = discs
        energy_dict[i] = energies

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
    grouper = LocalHourOfDayGrouper(
        time_col=tcol,
        tz_col=pcols["timezone_col"],
        start_time=pd.Timestamp(0),
        end_time=pd.Timestamp(0) + pd.Timedelta(params_slice["slice_freq"]),
        possible_tzs=params_summ["possible_tzs"],
    )
    profs = grouper.add_group_classes(profs)
    grp_cnts_tz = grouper.get_possible_obs_counts()
    n_bootstraps = profs[params_slice["bootstrap_id_col"]].nunique()
    grp_cnts_tz = grp_cnts_tz * n_bootstraps
    grp_cnts_tz = grp_cnts_tz.reset_index()
    grp_cnts_tz[tcol] = grp_cnts_tz[tcol].dt.tz_localize(None)
    grp_merge_cols = [pcols["timezone_col"]] + grouper.time_group_cols
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
