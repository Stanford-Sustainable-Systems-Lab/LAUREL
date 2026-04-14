"""Kedro pipeline nodes for the ``evaluate_impacts`` pipeline (Model Modules 5 & 6).

This module implements the final two model modules described in Passow & Rajagopal
(2026): estimating *expected* electrified dwell counts per location and assembling
per-substation/county charging load profiles across a fleet scaled to a target
adoption level.

Pipeline overview
-----------------
**Module 5 — Estimate expected electrified dwells:**

1.  **summarize_vehicles** — compute per-vehicle performance metrics (range
    deaths, charging delays) and flag vehicles whose simulation results are
    implausible for inclusion in load profiles.
2.  **apply_delays** — shift dwell start/end timestamps and update dwell
    durations to account for charging-induced delays.
3.  **filter_dwells_pre_prob** — retain only weekday dwells; keep zero-charge
    and non-electrified-vehicle dwells (needed for probability estimation).
4.  **filter_locs_pre_prob** — drop locations with missing required fields.
5.  **build_class_frame** — enumerate the full cross-product of vehicle and
    location class combinations.
6.  **compute_class_dwell_counts** — count observed dwells per
    (vehicle class × location class × electrification status).
7.  **compute_adoption_totals** — derive electrified-vehicle counts per vehicle
    class from scenario adoption forecasts; optionally override with scenario
    parameters.
8.  **compute_dwell_rate_vclass** — compute dwell rate (dwells/day) per vehicle
    class from observed data.
9.  **compute_class_probs** — fuse observed dwell location distributions with
    adoption forecasts to estimate P(electrified | location class, vehicle class)
    and expected dwell counts per class via logistic regression + correction term
    (``ElectProbLocalizer``).

**Module 6 — Assemble load profiles:**

10. **filter_dwells_post_prob** — drop dwells from non-electrified vehicles.
11. **add_dwell_id** — assign a sequential integer ID to each dwell row for
    joining with charging events.
12. **get_dwells_nonzero** — drop zero-charge dwells (creates a new DwellSet
    view, not in-place).
13. **manage_charging** — convert dwell records to charging event records using
    the configured charging manager.
14. **slice_events** — partition charging events into time-of-day windows and
    synthesise initialisation events for carry-over charging.
15. **build_time_ordered_slice** — sort and difference profile columns within
    each dwell to produce per-time-step power increments.
16. **sample_profiles_node** — core bootstrap sampling: build sparse
    correspondence matrices, compute inverse propensity weights, draw
    ``n_bootstraps`` samples, and return per-region load profiles and summaries.
17. **build_eval_columns** — inject ``group_cols`` into the shared ``pcols``
    parameter dictionary.
18. **add_region_geoms** — dissolve H3 hexagon correspondences into region
    polygons and join geometries to results.
19. **localize_time_from_hexes** — look up time zones from H3 hex centroids
    and convert UTC timestamps to local time.
20. **calc_utilization** — compute charging utilisation (energy / peak × window
    duration) for each profile type.
21. **compress_bootstrap_profiles** — discretise time and quantile-compress
    bootstrap profile draws into a compact representation.
22. **compress_bootstrap_summaries** — quantile-compress and mean-aggregate
    bootstrap scalar summaries.

Key design decisions
--------------------
- **Critical-day / zero-charge retention through Module 5**: Zero-charge dwells
  and dwells from non-electrified vehicles are retained through the probability
  estimation step because they contribute to observed dwell distributions that
  inform the logistic regression.  They are dropped only after class
  probabilities have been computed.
- **Inverse propensity score weighting**: Observed dwells are re-weighted by the
  reciprocal of P(observed & electrified | location, vehicle class) so that
  locations with low observation propensity are up-weighted when constructing
  load profiles.  This corrects for sampling bias in the telematics data.
- **Two-stage bootstrap sampling**: Each bootstrap draw first allocates expected
  dwell counts to hexagons (stage 1), then samples charging-event profiles from
  the pool of observed events at that hex or its freight-activity class (stage
  2).  The ``sample_self`` / ``sample_class`` flags control whether a hex can
  donate from its own observations, from class-level observations, or both.
- **Dask for bootstrap parallelism**: When ``n_bootstraps > 1``, individual
  draws are distributed across Dask workers via ``client.submit``; large numpy
  arrays are scattered with ``broadcast=True`` to avoid repeated serialisation.
- **Timedelta columns**: Several delay columns are stored as floats (hours) in
  the dwell data.  ``apply_delays`` converts them to ``timedelta`` for
  arithmetic and drops the temporary columns afterwards.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform
proactive substation upgrades for charging electric heavy-duty trucks.
*Applied Energy* (submitted March 2026).
"""

from __future__ import annotations

import gc
import itertools
import logging

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed
from tqdm.auto import tqdm

from laurel.models.dwell_sets import DwellSet
from laurel.models.group_times import AdaptiveTimeGrouper
from laurel.models.manage_charging import _MANAGER_MAP, ProfileType
from laurel.models.probability_localization import (
    ElectProbLocalizer,
    ElectProbLocalizerConfig,
)
from laurel.models.sampling import (
    build_entity_mask_array,
    normalize_sparse,
    sample_profiles,
)
from laurel.models.summarize import IntervalBeginSpreader, NonzeroGroupedSummarizer
from laurel.utils.data import filter_by_vals_in_cols
from laurel.utils.h3 import cells_to_region_polygons
from laurel.utils.params import set_entity_params
from laurel.utils.time import (
    calc_local_time,
    calc_time_zones_from_hexes,
    get_total_time_units_filtered,
    total_hours,
    total_time_units,
)

logger = logging.getLogger(__name__)


# ruff: noqa: PLR0915
def summarize_vehicles(dw: DwellSet, vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Compute per-vehicle performance metrics and flag implausible simulation results.

    Evaluates whether each vehicle's simulated electrification is operationally
    plausible by measuring two failure modes:

    1. **Range deaths** — trips where the vehicle ran out of charge
       (``dead_energy_col < 0``).  Circle trips (where the origin and
       destination hexagon are identical) are separated as non-addressable
       deaths that do not indicate a range design problem.
    2. **Charging delays** — extra time accumulated because the vehicle had to
       wait for charging.  Shift-level delay is computed as the cumulative delay
       at the end of the shift minus the delay at the start, plus any delay
       added at the preceding refresh stop.  Vehicle-days where simulation
       artefacts caused delay to drop (e.g. after a death event) are clipped to
       zero.

    Metrics are first aggregated by shift, then summarised to the vehicle level
    as point statistics (counts, percentages) and distributional statistics
    (quantiles across shifts).

    Vehicles that exceed any of the three thresholds — death rate, relative
    delay fraction, or absolute maximum delay — are flagged for exclusion from
    load-profile assembly.

    Args:
        dw: Post-simulation dwell dataset.  Must contain energy, delay, dwell
            duration, and refresh columns specified in ``params``.
        vehs: Vehicle-level table to augment with summary statistics and
            inclusion flags.  Returned with new columns appended.
        params: Pipeline parameters. Expected keys:

            - ``dead_energy_col`` (str): Column holding remaining energy after
              each trip; negative values indicate a death event.
            - ``charge_energy_col`` (str): Column holding energy charged at
              each dwell; non-NaN values indicate resuscitation after a death.
            - ``dwell_dur_col`` (str): Column holding dwell duration (hours).
            - ``refresh_col`` (str): Boolean column marking refresh dwells
              (used to delineate shift boundaries).
            - ``delay_inc_hrs_col`` (str): Column holding the delay increment
              added at each dwell.
            - ``delay_hrs_col`` (str): Column holding cumulative delay (hours).
            - ``summary_cols`` (dict): Mapping from logical names to output
              column names for shift-level metrics (e.g. ``max_delay``,
              ``shift_delay``, ``shift_dur``, ``shift_dur_delayed``,
              ``shift_dur_delayed_thresh``, ``delay_frac``).
            - ``shift_max_dur_hrs`` (float): Threshold shift duration (hours)
              above which a delayed shift is considered to violate operational
              constraints.
            - ``quantiles`` (list[float]): Quantile levels for distributional
              summaries.
            - ``delay_frac_thresh_quantile`` (float): Quantile of the delay
              fraction distribution to report and use for thresholding.
            - ``max_delay_thresh_quantile`` (float): Quantile of the max-delay
              distribution to report and use for thresholding.
            - ``thresholds`` (dict): Exclusion thresholds with sub-keys
              ``pct_shifts_w_deaths_max`` (float), ``delay_frac_max`` (float),
              and ``delay_hrs_max`` (float).
            - ``death_rate_col`` (str): Column name for the death-rate statistic
              used in the threshold check.
            - ``drop_events_col`` (str): Output boolean column marking vehicles
              to exclude from load profiles.
            - ``electrified_col`` (str): Output boolean column (inverse of
              ``drop_events_col``).

    Returns:
        The ``vehs`` DataFrame with per-vehicle summary statistics, quantile
        distribution columns, and two boolean flag columns (``drop_events_col``
        and ``electrified_col``) appended.
    """
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
    diecol = params["death_rate_col"]
    vehs["dies_too_freq"] = vehs[diecol] > thrs["pct_shifts_w_deaths_max"]
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
    """Apply charging-induced delays to dwell timestamps and durations.

    The charging-choice simulation records delays as floating-point hours.
    This node converts those columns to ``timedelta`` objects, then applies
    them to the dwell table so that downstream time-of-day analysis uses
    realised (delayed) times rather than originally scheduled times.

    Three adjustments are made to each dwell:

    1. ``dw.start`` is shifted forward by the cumulative delay accumulated up
       to the start of this dwell.
    2. Net dwell duration is updated as:
       ``original_duration − delay_decrease + delay_increase``.
    3. ``dw.end`` is shifted forward by the cumulative delay *plus* any new
       delay added at this dwell (``delay_increase``).

    Temporary ``timedelta`` columns are dropped before returning.

    Args:
        dw: Dwell dataset with float delay columns (hours). Modified in-place
            and returned.
        params: Pipeline parameters. Expected keys:

            - ``delay_columns`` (dict): Maps logical delay-role keys to column
              names in ``dw.data``.  Required keys: ``cumul_hrs`` (cumulative
              delay at dwell start), ``dwell_hrs`` (original dwell duration),
              ``decrease_hrs`` (delay reduction at this dwell), and
              ``increase_hrs`` (new delay added at this dwell).
            - ``delay_unit`` (str): Time unit of the float delay columns (e.g.
              ``"h"`` for hours), passed to ``pd.to_timedelta``.

    Returns:
        The updated ``DwellSet`` with adjusted ``dw.start``, ``dw.end``, and
        dwell duration column; temporary timedelta columns removed.
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
    """Filter dwells to the subset used for probability estimation.

    Retains dwells that fall within the desired time spans (optionally
    weekdays only) and drops rows with missing values in required columns.

    Critically, this filter does **not** drop:

    - Zero-charge dwells — they contribute to the observed dwell location
      distribution used by the logistic regression in ``compute_class_probs``.
    - Dwells from non-electrified vehicles — they are needed to estimate
      P(electrified | location class, vehicle class).

    These rows are only removed in the later ``filter_dwells_post_prob`` step,
    after class probabilities have been computed.

    Args:
        dw: Dwell dataset to filter. Modified in-place and returned.
        params: Pipeline parameters. Expected keys:

            - ``filter_out_weekends`` (bool): If ``True``, drop dwells whose
              start *and* end both fall on Saturday or Sunday.
            - ``drop_na_cols`` (list[str] | None): Column names for which rows
              with NaN values should be dropped. Pass ``None`` to skip.

        pcols: Shared pipeline column-name dictionary (not used directly in
            this node but required by the Kedro pipeline signature for
            consistency with sibling nodes).

    Returns:
        The filtered ``DwellSet`` with out-of-scope and missing-value rows
        removed.
    """
    if not dw.is_dask:
        old_len = len(dw.data)

    # Dwells which have at least some time within our time spans of interest
    if params["filter_out_weekends"]:
        is_weekday = _is_weekday(dw.data[dw.start]) | _is_weekday(dw.data[dw.end])
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


def _is_weekday(tser: pd.Series[pd.Timestamp]) -> pd.Series[bool]:
    """Return a boolean mask that is ``True`` for weekday (Mon–Fri) timestamps."""
    FIRST_DAY_OF_WEEKEND = 5
    is_weekday = tser.dt.weekday < FIRST_DAY_OF_WEEKEND
    return is_weekday


def filter_locs_pre_prob(locs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Drop locations that are missing required fields before probability estimation.

    Args:
        locs: Locations table (one row per H3 hexagon or reporting unit).
        params: Pipeline parameters. Expected keys:

            - ``drop_na_cols`` (list[str]): Column names for which rows with
              NaN values should be dropped (e.g. missing freight-activity
              class assignment).

    Returns:
        The filtered locations table.
    """
    locs_report = locs.dropna(subset=params["drop_na_cols"])
    return locs_report


def get_unique_series(df: pd.DataFrame, col: str, dropna: bool = True) -> pd.Series:
    """Extract the unique sorted values of a column or index level from a DataFrame.

    Args:
        df: Source DataFrame to query.
        col: Name of a column or index level whose unique values to extract.
        dropna: If ``True`` (default), remove NaN values from the result.

    Returns:
        A sorted ``pd.Series`` of unique values with ``name`` set to ``col``.

    Raises:
        RuntimeError: If ``col`` is not found in ``df.columns`` or
            ``df.index.names``.
    """
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
    """Build a complete cross-product of vehicle and location class combinations.

    Creates a DataFrame with one row for every (vehicle class × location class)
    pair, regardless of whether any observed dwells exist for that combination.
    This ensures that downstream probability and count computations include
    unobserved class pairs as explicit zeros rather than missing rows.

    Args:
        locs: Locations table containing location-class columns.
        vehs: Vehicle table containing vehicle-class columns.
        params: Pipeline parameters. Expected keys:

            - ``veh_class_cols`` (list[str]): Column names in ``vehs`` that
              define vehicle classes (e.g. primary operating distance class).
            - ``loc_class_cols`` (list[str]): Column names in ``locs`` that
              define location classes (e.g. freight-activity cluster).

    Returns:
        A DataFrame with one row per class combination and one column per
        class dimension.
    """
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
    """Count observed dwells per (vehicle class × location class × electrification).

    Groups dwell rows by all class columns plus the electrification flag, counts
    rows in each group, and pivots electrification into two columns
    (``n_dwells_electrified``, ``n_dwells_obs``).  The counts are left-joined
    onto the full class cross-product so that unobserved combinations appear
    as zeros.

    Args:
        classes: Full class cross-product DataFrame produced by
            ``build_class_frame``.
        dw: Dwell dataset that includes vehicle-class, location-class, and
            electrification columns.
        params: Pipeline parameters. Expected keys:

            - ``veh_class_cols`` (list[str]): Vehicle-class column names.
            - ``loc_class_cols`` (list[str]): Location-class column names.
            - ``electrified_col`` (str): Boolean column in ``dw.data``
              indicating whether the vehicle is electrified.

    Returns:
        The ``classes`` DataFrame with two new integer columns:
        ``n_dwells_obs`` (total observed dwells) and
        ``n_dwells_electrified`` (observed dwells from electrified vehicles).
        Unobserved combinations are filled with 0.
    """
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


def compute_adoption_totals(
    adopts: pd.DataFrame,
    params: dict,
    pcols: dict,
) -> pd.DataFrame:
    """Derive the number of electrified vehicles per class from adoption forecasts.

    Filters the adoption dataset to the rows of interest, groups by vehicle
    class, and sums electrified and total vehicle counts.  If
    ``override_with_params`` is enabled, the adoption fractions from the
    scenario parameter file replace those derived from the forecast data (while
    the total fleet size is preserved), allowing sensitivity analysis without
    re-running the full forecast pipeline.

    Args:
        adopts: Adoption forecast DataFrame with vehicle counts by fuel type
            and vehicle class.
        params: Pipeline parameters. Expected keys:

            - ``filter_totals`` (dict): Column-value pairs used to subset
              ``adopts`` before aggregation (e.g. year, region).
            - ``totals_column`` (str): Column holding vehicle counts.
            - ``fuel_type_col`` (str): Column identifying fuel type.
            - ``electrified_fuel_types`` (list[str]): Fuel type values
              considered electrified (e.g. ``["BEV"]``).
            - ``override_with_params`` (bool): If ``True``, replace observed
              adoption fractions with those in ``adoption_fracs``.
            - ``adoption_fracs`` (dict): Scenario-specific adoption fractions
              by vehicle class, used when ``override_with_params`` is ``True``.

        pcols: Shared column-name dictionary. Expected keys:

            - ``veh_class_cols`` (list[str]): Vehicle-class column names used
              for grouping.
            - ``electrified_col`` (str): Name for the electrification boolean
              column.

    Returns:
        A DataFrame indexed by vehicle class with columns for total fleet size
        and number of electrified vehicles.
    """
    adopts_sel = filter_by_vals_in_cols(adopts, params["filter_totals"])

    tot_col = params["totals_column"]
    adopts_sel.loc[:, tot_col] = adopts_sel[tot_col].astype(int)

    elect_col = pcols["electrified_col"]
    ftype_col = params["fuel_type_col"]
    adopts_sel[elect_col] = adopts_sel[ftype_col].isin(params["electrified_fuel_types"])
    adopt_grper = pcols["veh_class_cols"] + [elect_col]
    adopt_elect = adopts_sel.groupby(adopt_grper)[tot_col].sum()
    adopt_elect = adopt_elect.unstack(elect_col)
    elect_tot_col = f"n_vehs_{elect_col}"
    renamer = {True: elect_tot_col, False: f"n_vehs_not_{elect_col}"}
    adopt_elect.columns = [renamer[old] for old in adopt_elect.columns]

    adopt_elect[tot_col] = adopt_elect.sum(axis=1)
    adopt_elect = adopt_elect.drop(columns=[renamer[False]])

    # If desired, override the adoption fractions (but not the total vehicles of all fuel types)
    if not params["override_with_params"]:
        return adopt_elect
    else:
        acol = "adoption_fracs"
        pdict = {acol: params[acol]}
        adopts_over = set_entity_params(adopt_elect, pdict)
        adopts_over[elect_tot_col] = adopts_over[tot_col] * adopts_over[acol]
        adopts_over[elect_tot_col] = adopts_over[elect_tot_col].round().astype(int)
        adopts_over = adopts_over.drop(columns=[acol])
        return adopts_over


def compute_dwell_rate_vclass(
    veh_classes: pd.DataFrame,
    dw: DwellSet,
    vehs: pd.DataFrame,
    params: dict,
    pcols: dict,
) -> pd.DataFrame:
    """Compute the observed dwell rate (dwells per unit time) by vehicle class.

    Counts the total number of dwells per vehicle, divides by each vehicle's
    observation window length (in the configured time unit, optionally
    excluding weekends), aggregates to the vehicle-class level, and joins the
    resulting rate back onto the vehicle-class cross-product frame.

    The rate is used by ``compute_class_probs`` to scale expected dwell counts
    from per-vehicle-per-time-unit rates to absolute expected dwell counts
    across the target fleet.

    Args:
        veh_classes: Vehicle-class cross-product frame to augment.
        dw: Dwell dataset (only dwell counts per vehicle are needed).
        vehs: Vehicle table containing observation start/end timestamps and
            vehicle-class columns.
        params: Pipeline parameters. Expected keys:

            - ``dwell_count_col`` (str): Temporary column name for per-vehicle
              dwell counts.
            - ``obs_dur_cols`` (dict): Maps ``"start"`` and ``"end"`` to the
              column names in ``vehs`` holding observation window timestamps.
            - ``time_unit`` (str): Time unit for the observation window
              (e.g. ``"day"``).
            - ``filter_out_weekends`` (bool): If ``True``, exclude weekend
              days from each vehicle's observation window length.
            - ``dwell_rate_col`` (str): Output column name for the dwell rate
              (dwells per time unit).

        pcols: Shared column-name dictionary. Expected key:

            - ``veh_class_cols`` (list[str]): Vehicle-class column names.

    Returns:
        The ``veh_classes`` DataFrame with new columns for total dwell count,
        total observation time, vehicle count, and dwell rate appended.
    """
    dw_count_col = params["dwell_count_col"]
    vclass_cols = pcols["veh_class_cols"]
    odur_cols = params["obs_dur_cols"]

    n_dwells = dw.data.groupby(dw.veh).size()
    n_dwells.name = dw_count_col
    n_dwells = n_dwells.to_frame()
    mrg = vehs.loc[:, list(odur_cols.values()) + vclass_cols]
    n_dwells = n_dwells.merge(mrg, how="right", left_index=True, right_index=True)

    t_unit = params["time_unit"]
    obs_t_units_col = f"obs_{t_unit}_units"

    if params["filter_out_weekends"]:
        filterer = _is_weekday
    else:
        filterer = None

    n_dwells[obs_t_units_col] = n_dwells.apply(
        lambda row: get_total_time_units_filtered(
            start=row[odur_cols["start"]],
            end=row[odur_cols["end"]],
            unit=t_unit,
            filterer=filterer,
        ),
        axis=1,
    )
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
    """Estimate electrification probability and expected dwell counts per class.

    This is the core data-fusion step of Module 5.  It combines observed dwell
    location distributions from the telematics dataset with electrification
    adoption forecasts to produce, for every (location class, vehicle class)
    combination:

    1. **P(L|V)** — probability of a dwell falling in location class *L* given
       vehicle class *V*, estimated from observed dwell counts.
    2. **P(L,V)** — joint probability of a dwell falling in class pair (L, V),
       estimated from observed dwell counts.
    3. **P(E|V)** — probability of a vehicle being electrified given vehicle
       class *V*, derived from the adoption forecast.
    4. **P(E|L,V)** — probability of electrification given both location and
       vehicle class, estimated by ``ElectProbLocalizer`` (logistic regression
       + correction term that reconciles the class-level P(E|V) with location-
       specific dwell observations).
    5. **Expected electrified dwells** — ``P(L|V) × dwell_rate × n_vehs_in_class
       × P(E|L,V)``, representing the expected number of electrified dwells per
       time unit at each class combination.
    6. **P(observed & electrified | L,V)** — inverse propensity weight
       numerator used downstream to correct for observation bias.

    Args:
        cls: Class-level DataFrame containing dwell counts (from
            ``compute_class_dwell_counts``), fleet totals (from
            ``compute_adoption_totals``), and dwell rates (from
            ``compute_dwell_rate_vclass``), joined together.
        params: Pipeline parameters. Expected keys:

            - ``loc_class_cols`` (list[str]): Must contain exactly one element
              (multi-column location classes are not yet supported).
            - ``veh_class_cols`` (list[str]): Must contain exactly one element.
            - ``columns`` (dict): Maps logical names to column names in
              ``cls``, including ``n_dwells_obs``, ``n_dwells_obs_elect``,
              ``n_vehs_in_class``, ``n_vehs_electrified_in_class``, and
              ``dwell_rate``.
            - ``out_columns`` (dict): Output column name mapping with keys
              ``prob_obs_elect`` and ``n_elect_dwells_expected``.

    Returns:
        A DataFrame indexed by (location class, vehicle class) with columns
        for ``prob_obs_elect`` (inverse propensity weight numerator) and
        ``n_elect_dwells_expected`` (expected electrified dwell count per
        class per unit time).

    Raises:
        ValueError: If ``loc_class_cols`` or ``veh_class_cols`` contains more
            than one element.
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

    # Get number of dwells expected by class combination (\Delta_{C(h), k}).
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
    """Retain only dwells from vehicles flagged as electrified.

    Applied after class probabilities have been computed.  Non-electrified
    vehicle dwells — which were retained through ``filter_dwells_pre_prob``
    to inform probability estimation — are now dropped because they should
    not contribute to load profiles.

    Zero-charge dwells from electrified vehicles are still retained here;
    they are needed for the dwell-based sampling in ``sample_profiles_node``
    (a zero-charge dwell at a location still informs that location's dwell
    count) and are only removed in ``get_dwells_nonzero``.

    Note: This function returns a *new* ``DwellSet`` (``copy_without_data``),
    not an in-place modification of ``dw``.

    Args:
        dw: Full dwell dataset including non-electrified vehicle rows.
        pcols: Shared column-name dictionary. Expected key:

            - ``electrified_col`` (str): Boolean column marking electrified
              vehicles (produced by ``summarize_vehicles``).

    Returns:
        A new ``DwellSet`` containing only rows from electrified vehicles,
        with the ``electrified_col`` column dropped.
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
    """Assign a sequential integer dwell ID to each row.

    The ID is used as a join key between the dwell table and the charging
    events table produced by ``manage_charging``.

    Args:
        dw: Dwell dataset. Modified in-place and returned.
        params: Pipeline parameters. Expected key:

            - ``dw_col`` (str): Name for the new integer ID column.

    Returns:
        The updated ``DwellSet`` with a new sequential integer ID column.
    """
    dw.data[params["dw_col"]] = np.arange(len(dw.data))
    return dw


def get_dwells_nonzero(dw: DwellSet, pcols: dict) -> DwellSet:
    """Return a new DwellSet containing only dwells with nonzero charging.

    Zero-charge dwells were retained through the probability-estimation steps
    because they contribute to observed dwell location distributions.  They
    are dropped here because a sampled dwell with zero charge does not
    contribute energy to a load profile even if it is selected during
    bootstrap sampling.

    This function creates a *new* ``DwellSet`` view rather than modifying
    ``dw`` in-place, so the original (with zero-charge rows) remains available
    if needed.

    Args:
        dw: Dwell dataset containing both zero-charge and nonzero-charge rows.
        pcols: Shared column-name dictionary. Expected key:

            - ``charge_col`` (str): Column holding simulated charge amount
              (kWh); rows where this is ``<= 0`` are excluded.

    Returns:
        A new ``DwellSet`` containing only rows where charge amount is
        positive.
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
    """Convert dwell records into time-resolved charging event records.

    Instantiates the configured charging manager (selected from ``_MANAGER_MAP``
    by ``params["charging_manager"]``) and calls ``get_events()`` to expand
    each dwell into one or more charging events with start time, duration, and
    power columns.  Power values are rounded to reduce output file size.

    Args:
        dw: Dwell dataset with charge amounts, dwell timings, and vehicle
            parameters required by the charging manager.
        params: Pipeline parameters. Expected keys:

            - ``charging_manager`` (str): Key into ``_MANAGER_MAP`` selecting
              the charging manager class (e.g. ``"constant_power"``).
            - ``input_cols`` (dict): Column-name mappings forwarded to the
              charging manager constructor.
            - ``round_decimals`` (int): Number of decimal places to which
              power values are rounded.

    Returns:
        A DataFrame of charging events, with one row per time step per dwell,
        containing start time, duration, and power columns.
    """
    manager_cls = _MANAGER_MAP[params["charging_manager"]]
    manager = manager_cls(
        dw=dw, **params["input_cols"], prof_type=ProfileType.OBSERVATIONS
    )
    events = manager.get_events()
    pow_col = manager_cls.suffixes["power"]
    events[pow_col] = events[pow_col].round(params["round_decimals"])
    return events


def slice_events(events: pd.DataFrame, params: dict, pcols: dict) -> pd.DataFrame:
    """Partition charging events into time-of-day windows and handle cross-window carry-over.

    A "slice" is a fixed-frequency time window (e.g. one day) within which
    charging load profiles are assembled.  Events that begin in one window
    may extend into the next; this function handles that carry-over by
    synthesising initialisation events at the start of each new window.

    The algorithm:

    1. Optionally clip event durations to the window frequency to preserve
       the characteristic shape of a charging session without allowing a
       single event to dominate multiple windows.
    2. Use ``IntervalBeginSpreader`` to detect all events that cross a window
       boundary and generate initialisation rows at each affected window
       start.
    3. Concatenate original events with initialisation rows, assign each row
       to its window (``slice_id_col``), and compute each event's
       within-window start time (``slice_time_col``) as a timezone-naive
       offset from midnight.

    Args:
        events: Charging events DataFrame from ``manage_charging``.
        params: Pipeline parameters. Expected keys:

            - ``slice_id_col`` (str): Output column for the window identifier
              (floored timestamp of the window start).
            - ``slice_time_col`` (str): Output column for within-window
              offset (timezone-naive ``Timestamp`` relative to epoch).
            - ``source_time_col`` (str): Existing column holding each event's
              absolute start timestamp (must be timezone-naïve).
            - ``slice_freq`` (str): Pandas frequency string defining the
              window size (e.g. ``"1D"`` for one day).
            - ``clip_dur_to_slice_freq`` (bool): Whether to clip event
              durations to the window frequency before spreading.

        pcols: Shared column-name dictionary. Expected keys:

            - ``dw_col`` (str): Dwell ID column used as grouping key.
            - ``duration_col`` (str): Event duration column.
            - ``profile_cols`` (dict): Profile value columns to carry over
              into initialisation rows.

    Returns:
        A DataFrame of events (original + initialisation rows) with
        ``slice_time_col`` added and ``source_time_col`` /
        ``slice_id_col`` dropped.

    Raises:
        RuntimeError: If ``source_time_col`` contains timezone-aware
            timestamps.
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
    """Convert cumulative profile columns into per-time-step power increments.

    ``manage_charging`` produces cumulative profile values (e.g. cumulative
    kWh delivered since the start of the charging session).  The bootstrap
    sampler in ``sample_profiles_node`` needs *differences* — the incremental
    kWh delivered in each time step — to correctly aggregate contributions
    from multiple dwells.

    Sorts events by dwell ID and within-window time, computes first-order
    differences of each profile column within each dwell group (using the
    first row's absolute value for the initial diff), and drops the original
    cumulative columns.  The result is then sorted by within-window time to
    facilitate time-aligned aggregation during sampling.

    Args:
        events: Sliced events DataFrame from ``slice_events``.
        params: Pipeline parameters. Expected key:

            - ``slice_time_col`` (str): Column holding within-window time
              offset, used as the secondary sort key.

        pcols: Shared column-name dictionary. Expected keys:

            - ``dw_col`` (str): Dwell ID column used for grouping.
            - ``profile_cols`` (dict): Maps logical profile keys to cumulative
              column names (input).
            - ``diff_cols`` (dict): Maps the same logical keys to output
              difference column names; must be paired 1-to-1 with
              ``profile_cols``.

    Returns:
        A DataFrame sorted by ``slice_time_col`` with cumulative profile
        columns replaced by per-time-step difference columns.

    Raises:
        ValueError: If ``profile_cols`` and ``diff_cols`` do not have the
            same key set.
    """
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
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """Assemble per-region load profiles via inverse-propensity-weighted bootstrap sampling.

    This is the core computational node of Module 6.  For each bootstrap draw
    it:

    1. Builds four sparse correspondence matrices:

       - **Ga** (Dwells × Hexagons): indicates which hex each observed dwell
         occurred in.
       - **Cy** (Hexagons × Freight-activity classes): maps each hex to its
         freight-activity class.
       - **Be** (Events × Dwells): maps each charging event to its parent
         dwell.
       - **Rho** (Regions × Hexagons): maps each hex to its reporting region
         (substation or county).

    2. Computes inverse propensity weights **Ω** for each dwell using
       ``P(observed & electrified | location, vehicle class)`` from
       ``compute_class_probs``, normalised column-wise so that each hex (or
       class) receives unit total weight.

    3. Computes expected dwell counts per hex (``m_hex_expected``) by
       distributing class-level expected counts uniformly across hexes within
       each class.

    4. Calls ``sample_profiles`` for each bootstrap draw.  When
       ``n_bootstraps > 1``, draws are distributed across Dask workers; large
       arrays are scattered with ``broadcast=True`` to avoid redundant
       serialisation.

    5. Reduces each bootstrap profile DataFrame to one scalar per (region,
       time_bin) on the fly and accumulates into a compact dict, then
       returns confidence diagnostics (observed vs. expected hex dwell counts).

    Args:
        dw: Electrified, nonzero-charge dwell dataset with dwell IDs and hex
            columns.
        events: Time-ordered per-time-step charging event increments from
            ``build_time_ordered_slice``.
        locs: Locations table with hex IDs, region assignments, and
            freight-activity class codes.
        classes: Class-level probability and expected-count table from
            ``compute_class_probs``.
        params: Pipeline parameters. Expected keys:

            - ``sample_self`` (bool): Allow a hex to sample from its own
              observed dwells in stage 2 of the bootstrap.
            - ``sample_class`` (bool): Allow a hex to sample from class-level
              pooled dwells in stage 2.
            - ``n_bootstraps`` (int): Number of bootstrap draws (≥ 1).
            - ``loc_group_col`` (str): Column in ``locs`` holding the
              freight-activity class (categorical dtype required).
            - ``dwell_prob_obs_elect_col`` (str): Column in dwell data holding
              the inverse propensity weight numerator P(obs & elect | L, V).
            - ``n_dwells_expected_elect_col`` (str): Column in ``classes``
              holding expected electrified dwell counts per class.
            - ``max_first_stage_options`` (int): Maximum number of candidate
              dwell indices considered in stage 1 of sampling (caps memory use).
            - ``time_col`` (str): Within-window time column in ``events``.
            - ``slice_freq`` (str): Window frequency string (e.g. ``"1D"``).
            - ``discrete_freq`` (str): Discretisation frequency for profile
              compression.
            - ``summary_suffixes`` (dict): Suffix strings for cumulative and
              peak summary columns.
            - ``bootstrap_id_col`` (str): Column name for bootstrap draw ID
              in output.
            - ``master_seed`` (int | None): Random seed base; draw *i* uses
              ``master_seed + i`` if set.

        pcols: Shared column-name dictionary. Expected keys:

            - ``hex_col``, ``group_cols``, ``dw_col``, ``duration_col``,
              ``profile_cols``, ``diff_cols``, ``cum_cols``, ``peak_cols``.

        client: Dask distributed ``Client`` used to scatter data and submit
            bootstrap draws when ``n_bootstraps > 1``.

    Returns:
        A four-tuple of:

        - **boot_profs_accum** (``dict``): Compact accumulator mapping
          ``(region, time_bin)`` tuples to a list-of-lists of observed
          non-zero profile values across bootstrap draws, one inner list
          per profile column.  Passed directly to
          :func:`compress_bootstrap_profiles`.
        - **boot_summs** (``pd.DataFrame``): Per-bootstrap, per-region
          scalar summaries (cumulative energy, peak power).
        - **hex_confidence** (``pd.DataFrame``): Per-hexagon observed and
          expected dwell counts for diagnostic use.
        - **debug_partition** (``dict``): Empty dict when
          ``params["save_bootstrap_profiles"]`` is ``False`` (production).
          When ``True``, maps zero-padded bootstrap ID strings to the
          full per-bootstrap profile DataFrame captured before accumulation,
          for writing to the ``bootstrap_profiles_debug_partition`` catalog
          entry.

    Raises:
        ValueError: If ``pcols["group_cols"]`` has more than one element, or
            if a profile column name conflicts with an argument of
            ``sample_profiles``, or if ``n_bootstraps < 1``.
        NotImplementedError: If ``locs[loc_group_col]`` is not a categorical
            dtype.
    """
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

    if len(pcols["group_cols"]) > 1:  # TODO: Enable multi-column
        raise NotImplementedError(
            "Only a single grouping column is currently implemented."
        )
    reg_col = pcols["group_cols"][0]
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
    loc_groups = locs_counts[params["loc_group_col"]]
    if isinstance(loc_groups.dtype, pd.CategoricalDtype):
        class_arr = loc_groups.cat.codes.values
        class_map = {
            category: code for code, category in enumerate(loc_groups.cat.categories)
        }
    else:
        raise NotImplementedError("Only categorical dtypes currently supported.")
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
    m_hex_obs = Ga.T @ np.ones(shape=(len(dwells_samp),), dtype=np.int64)
    m_class_obs = Cy.T @ m_hex_obs
    locs_counts["m_hex_observed"] = m_hex_obs

    exp_dwell_rate_col = params["n_dwells_expected_elect_col"]
    loc_cls = classes.groupby(params["loc_group_col"], observed=False)[
        exp_dwell_rate_col
    ].sum()
    loc_cls.index = loc_cls.index.map(class_map)  # In case categories were scrambled
    loc_cls = loc_cls.sort_index()
    n_locs_per_class = Cy.sum(axis=0)
    m_class_expected = loc_cls.values / n_locs_per_class
    m_hex_expected = Cy @ m_class_expected
    locs_counts["m_hex_expected"] = m_hex_expected

    # Get profiles
    tcol = params["time_col"]
    diff_cols = list(pcols["diff_cols"].values())
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
        "event_times": events[tcol].values,
        "slice_freq": params["slice_freq"],
        "discrete_freq": params["discrete_freq"],
        "dur_col": pcols["duration_col"],
        "summary_suffixes": params["summary_suffixes"],
        "region_name": reg_col_compact,
        "time_col": tcol,
        "sample_self": params["sample_self"],
        "sample_class": params["sample_class"],
    }

    for diff_col in diff_cols:
        if diff_col in kws:
            raise ValueError(
                f"Profile column name '{diff_col}' is overriding an existing argument of sample_profiles."
            )
        kws.update({diff_col: events[diff_col].values})

    # Pre-compute rename lookups so they are available inside _accumulate.
    # profile_cols renames diff columns (e.g. power_kw_diff → power_kw).
    loc_counts_names = locs_counts.loc[:, [reg_col_compact, reg_col]].drop_duplicates()
    reg_name_restorer = loc_counts_names.set_index(reg_col_compact)[reg_col]
    prof_renamer = {d: pcols["profile_cols"][k] for k, d in pcols["diff_cols"].items()}
    value_cols = list(pcols["profile_cols"].values())
    # summ_cols defines the accumulator group key: (region, time_bin).
    # No AdaptiveTimeGrouper needed — discretize_sparse_profiles already aligns
    # timestamps to discrete_freq boundaries (one row per group guaranteed).
    summ_cols = pcols["group_cols"] + [tcol]

    logger.info("Perform bootstrap sampling")
    master_seed = params.get("master_seed", None)
    boot_summs = {}
    boot_profs_accum: dict = {}
    debug_partition: dict = {}
    save_debug = params.get("save_bootstrap_profiles", False)

    def _accumulate(prof: pd.DataFrame, summ: pd.DataFrame, boot_id: int) -> None:
        """Rename columns, optionally capture for debug, then reduce to accumulator.

        Applies column renaming and region-label restoration in place, captures
        the full DataFrame if debug output is enabled, then reduces to the
        accumulator using ``DataFrame.to_dict("index")``.  This produces a
        single C-level ``{group_key: {col: val}}`` mapping, avoiding the
        per-element boxing overhead of pandas Series iteration.
        ``discretize_sparse_profiles`` guarantees one row per (region,
        time_bin) per bootstrap, so no further groupby is needed.

        Args:
            prof: Raw profile DataFrame from ``sample_profiles``, using compact
                integer region codes and diff column names.
            summ: Raw summary DataFrame from ``sample_profiles``.
            boot_id: Zero-based bootstrap draw index, used as the debug
                partition key.
        """
        prof = prof.rename(columns=prof_renamer)
        prof[reg_col] = prof[reg_col_compact].map(reg_name_restorer)
        prof = prof.drop(columns=[reg_col_compact])
        if save_debug:
            debug_partition[str(boot_id).zfill(4)] = prof.reset_index(drop=True)
        rows_dict = prof.set_index(summ_cols)[value_cols].to_dict("index")
        for group_key, col_vals in rows_dict.items():
            if group_key not in boot_profs_accum:
                boot_profs_accum[group_key] = [[] for _ in value_cols]
            for col_idx, col in enumerate(value_cols):
                boot_profs_accum[group_key][col_idx].append(col_vals[col])
        boot_summs[boot_id] = summ

    if n_boots == 1:
        prof, summ = sample_profiles(**kws, seed=master_seed)
        _accumulate(prof, summ, boot_id=0)
    elif n_boots > 1:

        def _sample_profiles_distrib(kws: dict, boot_id: int):
            if master_seed is not None:
                seed = master_seed + boot_id
            else:
                seed = master_seed
            return sample_profiles(**kws, seed=seed)

        # Scatter shared read-only arrays to all workers once, then immediately
        # drop the main-process copies — workers hold the only live references.
        future_kws = {k: client.scatter(v, broadcast=True) for k, v in kws.items()}
        del kws, Om_hex, Om_cls, Be, Rho, Cy, Ga
        gc.collect()

        # Sliding-window submission: keep two tasks queued per worker.  A window
        # of 2× n_workers staggers finish times so the "thundering herd" of all
        # workers completing simultaneously is broken after the first cycle.
        # This also ensures workers have queued work available during the
        # main-thread _accumulate step, preventing idle gaps.  Peak unconsumed
        # memory is bounded at 2× n_workers results (≈ 6 for 3 workers).
        n_window = 2 * len(client.scheduler_info()["workers"])

        def _submit(bid: int):
            return client.submit(_sample_profiles_distrib, kws=future_kws, boot_id=bid)

        boot_ids = iter(range(n_boots))
        future_to_boot_id: dict = {}

        # Seed the window: submit the first n_window tasks upfront.
        for bid in itertools.islice(boot_ids, n_window):
            f = _submit(bid)
            future_to_boot_id[f] = bid

        # Consume results as they arrive; submit one new task per result consumed
        # so the window depth stays constant (one-in, one-out).
        ac = as_completed(future_to_boot_id, with_results=True)
        for future, result in tqdm(ac, total=n_boots):
            boot_id = future_to_boot_id.pop(future)
            prof, summ = result
            # Submit the next task immediately — before _accumulate — so workers
            # get new work without waiting for the main-thread accumulation step.
            next_id = next(boot_ids, None)
            if next_id is not None:
                new_f = _submit(next_id)
                future_to_boot_id[new_f] = next_id
                ac.add(new_f)  # register with the live iterator so it gets yielded
            _accumulate(prof, summ, boot_id)
            del result
            future.release()  # release scheduler's copy of the result immediately
    else:
        raise ValueError("Number of bootstraps must be >= 1.")

    gc.collect()

    boot_summs = pd.concat(boot_summs, names=[params["bootstrap_id_col"]], copy=False)
    boot_summs = boot_summs.reset_index()

    # Restore region names in summaries
    summ_renamer = {}
    suffs = params["summary_suffixes"]
    ditems = pcols["diff_cols"].items()
    summ_renamer.update(
        {f"{d}{suffs["cumul"]}": pcols["cum_cols"][k] for k, d in ditems}
    )
    summ_renamer.update(
        {f"{d}{suffs["peak"]}": pcols["peak_cols"][k] for k, d in ditems}
    )
    boot_summs = boot_summs.rename(columns=summ_renamer)
    boot_summs[reg_col] = boot_summs[reg_col_compact].map(reg_name_restorer)
    boot_summs = boot_summs.drop(columns=[reg_col_compact])

    # Report on confidence of sampling
    conf_cols = [pcols["hex_col"], "m_hex_observed", "m_hex_expected"]
    hex_confidence = locs_counts.loc[:, conf_cols]

    return boot_profs_accum, boot_summs, hex_confidence, debug_partition


def build_eval_columns(pcols: dict, group_cols: list) -> dict:
    """Inject the reporting group columns into the shared pipeline column dictionary.

    ``group_cols`` specifies the spatial aggregation level for load profiles
    (e.g. substation ID or county code).  It is passed separately in the Kedro
    pipeline so that the same node logic can produce substation-level or
    county-level outputs by swapping the parameter.

    Args:
        pcols: Shared column-name dictionary to update.
        group_cols: List of column names that define the reporting grouping
            (e.g. ``["substation_id"]``).

    Returns:
        The updated ``pcols`` dictionary with ``"group_cols"`` set to
        ``group_cols``.
    """
    pcols.update({"group_cols": group_cols})
    return pcols


def add_region_geoms(
    results: pd.DataFrame,
    hex_regions: pd.DataFrame,
    params: dict,
) -> gpd.GeoDataFrame:
    """Dissolve H3 hex correspondences into region polygons and join to results.

    Converts a hex→region correspondence table into one polygon per region
    (by dissolving the H3 cell polygons), joins it to the results DataFrame,
    and converts any ``timedelta`` columns to float hours for serialisation
    compatibility.

    Args:
        results: Per-region summary or profile DataFrame to annotate with
            geometries.
        hex_regions: DataFrame mapping H3 hex IDs to region identifiers.
        params: Pipeline parameters. Expected keys:

            - ``hex_col`` (str): Column in ``hex_regions`` holding H3 hex IDs.
            - ``region_col`` (str): Column in both ``hex_regions`` and
              ``results`` holding the region identifier used to join.

    Returns:
        A ``GeoDataFrame`` with a ``geometry`` column (dissolved region
        polygons) and ``timedelta`` columns replaced by float-hours
        equivalents.
    """
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


def localize_time_from_hexes(
    df: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Convert UTC timestamps to local time using time zones inferred from H3 hexagons.

    Optionally looks up the IANA time zone for each row from the H3 hex
    centroid (via ``calc_time_zones_from_hexes``), then converts one or more
    UTC timestamp columns to local time (via ``calc_local_time``).  Daylight
    saving time ambiguities are introduced naturally by working with
    timezone-naïve local times.

    Args:
        df: DataFrame containing a hex column and UTC timestamp columns.
            Modified and returned.
        params: Pipeline parameters. Expected keys:

            - ``get_tz`` (bool): If ``True``, derive time zones from hex
              centroids and write them to ``pcols["timezone_col"]``.  Set to
              ``False`` if the time zone column is already present.
            - ``time_cols_source`` (list[str]): UTC timestamp columns to
              localise.
            - ``time_cols_local`` (list[str]): Output column names for the
              corresponding local-time timestamps.
            - ``sort_result`` (bool): If ``True``, sort ``df`` by
              ``sort_cols`` after localisation.
            - ``sort_cols`` (list[str]): Columns to sort by when
              ``sort_result`` is ``True``.

        pcols: Shared column-name dictionary. Expected keys:

            - ``hex_col`` (str): Column holding H3 hex IDs.
            - ``timezone_col`` (str): Column to store or read IANA time zone
              strings.

    Returns:
        The updated DataFrame with local-time columns added and, optionally,
        a time zone column added and rows sorted.
    """
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


def calc_utilization(summs: pd.DataFrame, params: dict, pcols: dict) -> pd.DataFrame:
    """Compute charging utilisation for each profile type from bootstrap summaries.

    Utilisation is defined as::

        utilisation = cumulative_energy / (peak_power × window_duration)

    where ``window_duration`` is derived from the slice frequency in the
    configured time unit.  A utilisation of 1.0 means the charger ran at peak
    power for the entire window; values below 1.0 reflect partial utilisation.

    Args:
        summs: Bootstrap summary DataFrame with cumulative-energy and
            peak-power columns for each profile type.
        params: Pipeline parameters. Expected keys:

            - ``slice_freq`` (str): Window frequency string used to compute
              total window duration (e.g. ``"1D"``).
            - ``dur_unit`` (str): Time unit for window duration conversion
              (e.g. ``"h"`` for hours).

        pcols: Shared column-name dictionary. Expected keys:

            - ``profile_cols`` (dict): Maps profile-type keys to column names
              (used to iterate over types).
            - ``util_cols`` (dict): Maps profile-type keys to output
              utilisation column names.
            - ``cum_cols`` (dict): Maps profile-type keys to cumulative-energy
              column names.
            - ``peak_cols`` (dict): Maps profile-type keys to peak-power
              column names.

    Returns:
        The ``summs`` DataFrame with one new utilisation column per profile
        type appended.
    """
    tot_dur = pd.Series([pd.Timedelta(params["slice_freq"])])
    tot_dur = total_time_units(tot_dur, unit=params["dur_unit"])
    tot_dur = tot_dur.values[0]

    for vtype in pcols["profile_cols"].keys():
        util_col = pcols["util_cols"][vtype]
        cum_col = pcols["cum_cols"][vtype]
        pk_col = pcols["peak_cols"][vtype]
        summs[util_col] = summs[cum_col] / (summs[pk_col] * tot_dur)
    return summs


def compress_bootstrap_profiles(
    profs: pd.DataFrame | dict, params: dict, pcols: dict
) -> pd.DataFrame:
    """Reduce bootstrap profile draws to quantile envelopes over discretised time.

    Accepts either a full bootstrap profile DataFrame (legacy path) or a
    compact accumulator dict produced by the memory-efficient path in
    :func:`sample_profiles_node`.  The two paths share the same output schema.

    **Accumulator path** (``profs`` is a ``dict``):
    The accumulator maps ``(region, time_bin)`` tuples to a list-of-lists of
    observed non-zero profile values.  Because timestamps are already at
    ``discrete_freq`` resolution (guaranteed by
    :func:`~laurel.models.sampling.discretize_sparse_profiles`), no
    time-bin re-assignment is needed.  ``n_bootstraps`` is read directly from
    ``params["n_bootstraps"]`` and used as the uniform possible-observation
    count for zero-padding.

    **DataFrame path** (``profs`` is a ``pd.DataFrame``):
    Unchanged from the original implementation — uses ``AdaptiveTimeGrouper``
    to assign time bins, computes per-bin possible-observation counts, and
    calls :meth:`NonzeroGroupedSummarizer.summarize`.  Retained for backward
    compatibility and as the fallback when multi-timezone DST correction is
    required.

    Args:
        profs: Either:

            - A ``pd.DataFrame`` with columns for region, time, bootstrap ID,
              and profile values (legacy path).
            - A ``dict`` mapping ``(region, time_bin)`` group keys to
              list-of-lists of observed values (accumulator path).

        params: Pipeline parameters. Expected keys:

            - ``time_col`` (str): Within-window time column.
            - ``discrete_freq`` (str): Coarsened time-bin frequency string.
            - ``slice_freq`` (str): Original window frequency string.
            - ``bootstrap_id_col`` (str): Bootstrap draw ID column
              (DataFrame path only).
            - ``n_bootstraps`` (int): Total bootstrap count used for
              zero-padding (accumulator path only).
            - ``quantiles`` (list[float]): Quantile levels to compute.

        pcols: Shared column-name dictionary. Expected keys:

            - ``timezone_col`` (str): Time zone column (DataFrame path only).
            - ``group_cols`` (list[str]): Region grouping columns.
            - ``profile_cols`` (dict): Maps profile-type keys to value column
              names to quantile.

    Returns:
        A DataFrame indexed by (region, time-bin) with quantile columns for
        each profile type.
    """
    value_cols = list(pcols["profile_cols"].values())

    if isinstance(profs, dict):
        # --- Accumulator path (memory-efficient) ---
        # n_bootstraps comes from params (already present in params:sample_profiles).
        # For the single-timezone (no_time_zone) case, possible_count is uniform
        # across all groups and equals n_bootstraps.
        n_bootstraps = params["n_bootstraps"]
        summ_cols = pcols["group_cols"] + [params["time_col"]]
        logger.info("Calculate quantiles (accumulator path)")
        summer = NonzeroGroupedSummarizer(
            group_cols=summ_cols,
            quantiles=np.array(params["quantiles"]),
        )
        return summer.summarize_from_accumulator(
            group_values=profs,
            n_possible=n_bootstraps,
            value_cols=value_cols,
        )

    # --- DataFrame path (original behaviour, unchanged) ---
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
    return summer.summarize(
        events=profs,
        value_cols=value_cols,
        possible_count_col="possible_count",
    )


def compress_bootstrap_summaries(
    summs: pd.DataFrame, params: dict, pcols: dict
) -> pd.DataFrame:
    """Reduce per-bootstrap scalar summaries to quantiles and means by region.

    Each bootstrap draw produces scalar summary statistics per region (total
    energy, peak power, utilisation).  This node compresses those draws to a
    compact set of distributional statistics — quantiles via
    ``NonzeroGroupedSummarizer`` and means — so that downstream analysis can
    reason about the uncertainty across bootstrap draws without storing all
    raw draws.

    Args:
        summs: Bootstrap summary DataFrame with columns for region, bootstrap
            ID, and scalar summary values (cumulative energy, peak power,
            utilisation).
        params: Pipeline parameters. Expected keys:

            - ``bootstrap_id_col`` (str): Column holding the bootstrap draw
              ID, used to determine ``n_bootstraps``.
            - ``quantiles`` (list[float]): Quantile levels to compute.

        pcols: Shared column-name dictionary. Expected keys:

            - ``group_cols`` (list[str]): Region grouping columns.
            - ``cum_cols`` (dict): Maps profile-type keys to cumulative-energy
              column names.
            - ``peak_cols`` (dict): Maps profile-type keys to peak-power
              column names.
            - ``util_cols`` (dict): Maps profile-type keys to utilisation
              column names.

    Returns:
        A DataFrame indexed by region with quantile columns (from
        ``NonzeroGroupedSummarizer``) and mean columns (suffixed ``_mean``)
        for each summary statistic.
    """
    n_bootstraps = summs[params["bootstrap_id_col"]].nunique()
    summs["possible_count"] = n_bootstraps

    logger.info("Calculate quantiles")
    summ_cols = pcols["group_cols"]
    summer = NonzeroGroupedSummarizer(
        group_cols=summ_cols,
        quantiles=np.array(params["quantiles"]),
    )

    val_cols = (
        list(pcols["cum_cols"].values())
        + list(pcols["peak_cols"].values())
        + list(pcols["util_cols"].values())
    )
    quantiles = summer.summarize(
        events=summs,
        value_cols=val_cols,
        possible_count_col="possible_count",
    )

    means = summs.groupby(summ_cols)[val_cols].mean()
    means.columns = [f"{col}_mean" for col in means.columns]

    comps = quantiles.join(means, how="outer")
    return comps
