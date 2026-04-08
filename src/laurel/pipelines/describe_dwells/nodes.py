"""Kedro pipeline nodes for the ``describe_dwells`` pipeline (Model Module 2 — Augment dwell data).

Transforms the raw telematics trip records into a clean, shift-annotated
``DwellSet`` that is ready for the electrification simulation.  This pipeline
implements the second part of Model Module 2 (Augment Dwell Data): coalescing
brief return trips that break a dwell without meaningfully changing the
vehicle's location, and marking the start of each new FMCSA-compliant driver
shift.

Pipeline overview
-----------------
1. **format_trips_columns** — Parses timestamps, converts H3 hex strings to
   integers, recodes the vehicle-ID column as a compact integer category,
   and optionally persists the Dask DataFrame in memory.
2. **calc_derived_trip_cols** — Computes trip duration in hours from start and
   end timestamps.
3. **create_dwells** — Converts the trip-oriented DataFrame into a
   ``DwellSet`` (one row per dwell event) using ``DwellSet.from_trips``.
4. **coalesce_interrupted_dwells** — Merges dwells that are separated by a
   short "circle trip" (same origin and destination, below distance and
   duration thresholds) into a single, longer dwell.
5. **mark_vehicle_shifts** — Marks dwell events that start a new driver shift
   (dwell duration >= ``min_refresh_hrs``, per FMCSA HOS regulations) and
   assigns a monotonically increasing shift ID within each vehicle.
6. **calc_rolling_dwell_ratios** — Computes a rolling time-window dwell ratio
   for each (vehicle, location) pair as the fraction of the vehicle's total
   dwell time spent at that location.
7. **map_location_groups** — Joins freight-activity-class labels from the
   ``describe_locations`` pipeline onto each dwell row.

Key design decisions
--------------------
- **Coalescing via ``accum_masked``**: Rather than a forward-fill or group-join,
  coalescing is implemented by propagating the latest end-time of the
  interrupted dwell sequence *backwards* through the masked rows and then
  dropping the non-masked rows.  This avoids a sort-dependent join and is
  compatible with Dask partitioned DataFrames.
- **Reset column neutralisation**: The ``DwellSet.reset`` column has special
  semantics inside ``accum_masked``.  Setting it to ``False`` before calling
  ``accum_masked`` and restoring it afterwards prevents the coalescing logic
  from treating existing shift boundaries as accumulation barriers.
- **FMCSA 6.9-hour threshold**: The ``min_refresh_hrs`` parameter encodes the
  FMCSA 8-hour off-duty rest requirement.  A slightly lower threshold of 6.9
  hours is used to accommodate GPS rounding in the telematics data.
- **Deprecated rolling-ratio functions**: ``calc_inter_visit_stats``,
  ``calc_inter_visit_times``, ``describe_veh_loc_pairs``,
  ``filter_substantial_dwells``, and ``cluster_veh_loc_pairs`` are preserved
  for potential future use but are not connected to the active pipeline as of
  2025-10-20.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.

Federal Motor Carrier Safety Administration. Hours of Service Regulations,
49 CFR Part 395.
"""

import logging
from copy import deepcopy

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from laurel.models.dwell_sets import CumAggFunc, DwellSet
from laurel.utils.h3 import str_to_h3
from laurel.utils.time import total_hours

logger = logging.getLogger(__name__)


def format_trips_columns(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Parse timestamps, encode H3 hex strings, and recode vehicle IDs in the raw trips DataFrame.

    Categorising the vehicle-ID column *before* converting timestamp columns
    prevents a known Dask issue in which ``dd.to_datetime`` silently converts
    integer category codes to ``float64``.  H3 hexagon strings are converted
    to ``uint64`` integers for memory efficiency and join performance.

    Args:
        trips: Raw trips Dask DataFrame as loaded from the ``01_raw`` catalog.
        params: Pipeline parameters dict with keys:

            - ``category_columns`` (list[str]): columns to categorise before
              timestamp conversion.
            - ``time_columns`` (list[str]): columns to convert to UTC-aware
              ``datetime64``.
            - ``h3_columns`` (list[str]): H3 hexagon string columns to convert
              to ``uint64``.
            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names (applied after all other transformations).
            - ``veh_id_col`` (str): vehicle identifier column; category codes
              are cast to ``int64`` after renaming.
            - ``persist`` (bool): if ``True``, trigger Dask computation and
              pin the result in distributed memory.

    Returns:
        A Dask DataFrame with parsed timestamps, integer hex IDs, compact
        integer vehicle IDs, and columns renamed to internal names.
    """
    trips = trips.categorize(params["category_columns"])

    for col in params["time_columns"]:
        # WARNING: This line somehow converts vehicle_id to a float64 from an int64,
        # to fix this, I'm categorizing the vehicle_id column first.
        trips[col] = dd.to_datetime(trips[col], utc=True)

    for col in params["h3_columns"]:
        trips[col] = trips[col].map_partitions(str_to_h3, meta=(col, "int"))

    trips = trips.rename(columns={v: k for k, v in params["col_renamer"].items()})
    trips[params["veh_id_col"]] = trips[params["veh_id_col"]].cat.codes.astype(np.int64)

    if params["persist"]:
        trips = trips.persist()

    return trips


def calc_derived_trip_cols(trips: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Compute trip duration in hours from start and end timestamp columns.

    Args:
        trips: Trips DataFrame with UTC-aware timestamp columns.
        params: Pipeline parameters dict with keys:

            - ``time_cols`` (dict): sub-keys ``trip_start`` and ``trip_end``
              naming the timestamp columns.
            - ``persist`` (bool): if ``True``, trigger Dask computation and
              pin the result in distributed memory.

    Returns:
        The trips DataFrame with a new ``trip_hrs`` column (float).
    """
    trips["trip_hrs"] = total_hours(
        trips[params["time_cols"]["trip_end"]]
        - trips[params["time_cols"]["trip_start"]]
    )

    if params["persist"]:
        trips = trips.persist()
    return trips


def create_dwells(trips: dd.DataFrame, params: dict, client: Client) -> DwellSet:
    """Convert a trip-oriented DataFrame into a ``DwellSet`` of dwell events.

    A dwell event represents the period during which a vehicle is stationary
    at a location between two consecutive trips.  ``DwellSet.from_trips``
    infers each dwell's start time, end time, and hexagon from the
    surrounding trip records.

    An optional debug subsample limits computation to the first ``n`` rows of
    the Dask DataFrame for rapid iteration.  If ``load_into_memory`` is set,
    the entire Dask DataFrame is materialised before conversion, which can be
    faster when working on smaller datasets.

    Args:
        trips: Trips Dask DataFrame (output of ``calc_derived_trip_cols``).
        params: Pipeline parameters dict with keys:

            - ``debug_subsample`` (dict): ``active`` (bool) and ``n`` (int)
              to limit the input to the first ``n`` rows.
            - ``load_into_memory`` (bool): if ``True``, call ``.compute()``
              before conversion.
            - ``drop_cols`` (list[str]): trip columns to remove before
              conversion.
            - ``col_renamer`` (dict[str, str]): additional column renames to
              apply before conversion.
            - ``from_trips_cols`` (dict): column-name keyword arguments
              forwarded to ``DwellSet.from_trips`` (e.g., ``veh``, ``hex``,
              ``start``, ``end``).
            - ``verify_sorting`` (bool): if ``True``, ``DwellSet.from_trips``
              verifies that trips are sorted by (vehicle, time).
            - ``set_index_kwargs`` (dict): additional keyword arguments for
              setting the index in ``DwellSet.from_trips``.
        client: Active Dask distributed ``Client`` (used implicitly by the
            Dask scheduler; not called directly in this function).

    Returns:
        A ``DwellSet`` with one dwell record per stationary event, sorted by
        (vehicle, start time).
    """
    if params["debug_subsample"]["active"]:
        trips = trips.loc[0 : params["debug_subsample"]["n"]]

    if params["load_into_memory"]:
        logger.info("Loading dataset into memory")
        trips = trips.compute()

    logger.info("Converting to dwells from trips.")
    drop_cols = list(set(params["drop_cols"]).intersection(trips.columns))
    trips = trips.drop(columns=drop_cols)
    trips = trips.rename(columns={v: k for k, v in params["col_renamer"].items()})
    colnames = params["from_trips_cols"]
    dw = DwellSet.from_trips(
        trips=trips,
        verify_sorting=params["verify_sorting"],
        **colnames,
        **params["set_index_kwargs"],
    )
    return dw


def coalesce_interrupted_dwells(dw: DwellSet, params: dict) -> DwellSet:
    """Merge dwells that are split by short circle trips into single, continuous dwells.

    A "circle trip" is a trip whose origin and destination hexagon are
    identical and whose distance and duration both fall below configurable
    thresholds.  Such trips most likely represent GPS noise or brief vehicle
    movements within a depot and should not break the surrounding dwell into
    two separate events.

    The algorithm proceeds as follows:

    1. Mark non-circle trips with ``is_not_short_circle = True``.
    2. Temporarily set ``dw.reset`` to ``False`` for all rows to prevent
       shift boundaries from interrupting the accumulation.
    3. Use ``DwellSet.accum_masked`` with ``CumAggFunc.MAX`` to propagate the
       *latest* end time backwards through each circle-trip gap (``reverse=True``),
       so that the dwell preceding the circle trip absorbs the end time of
       the dwell following it.
    4. Drop rows marked as circle trips.
    5. Restore ``dw.reset`` from the saved copy and rename accumulated columns
       back to their original names.

    The distances and durations of the dropped circle trips are not
    accumulated into the surviving dwell (they are treated as negligible).

    Args:
        dw: Input ``DwellSet`` freshly created from trips.
        params: Pipeline parameters dict with keys:

            - ``max_short_dist_miles`` (float): maximum trip distance (miles)
              for a trip to qualify as a circle trip.
            - ``max_short_dur_hrs`` (float): maximum trip duration (hours) for
              a trip to qualify as a circle trip.

    Returns:
        The ``DwellSet`` with circle trips coalesced; ``dw.data`` has the same
        schema as the input but with fewer rows.
    """
    prev_col = f"{dw.hex}_prev"
    mask_col = "is_not_short_circle"

    max_dist = params["max_short_dist_miles"]
    max_dur = params["max_short_dur_hrs"]

    logger.info("Setting shift id column")
    shift_kwargs = {"fill_value": 0}
    if dw.is_dask:
        shift_kwargs["meta"] = ("x", dw.data[dw.hex].dtype)
    dw.data[prev_col] = dw.data.groupby(dw.veh)[dw.hex].shift(1, **shift_kwargs)
    dw.data[mask_col] = ~(
        (dw.data[prev_col] == dw.data[dw.hex])
        & (dw.data[dw.trip_dist] < max_dist)
        & (dw.data[dw.trip_dur] < max_dur)
    )
    dw.data = dw.data.drop(columns=[prev_col])

    # Comments suggest some modifications to apply if distance and duration of dropped
    # trips should be retained.
    logger.info("Accumulating columns across coalescing dwells")
    reset_orig_col = f"{dw.reset}_original"
    dw.data[reset_orig_col] = dw.data[dw.reset].copy()
    dw.data[dw.reset] = (
        False  # Necessary to avoid the special reset-handling in accum_masked
    )
    accum_cols = [reset_orig_col, dw.end]  # "trip_miles", "trip_hrs", "dwell_hrs"
    agg_funcs = [
        CumAggFunc.MAX,
        CumAggFunc.MAX,
    ]  # CumAggFunc.SUM, CumAggFunc.SUM, CumAggFunc.SUM

    dw.data[dw.end] = dw.data[dw.end].astype(np.int64)
    dw.accum_masked(
        keep_mask_col=mask_col,
        accum_cols=accum_cols,
        reverse=True,
        agg_func=agg_funcs,
        write_all=False,
        inplace=True,
    )
    mod_col = f"{dw.end}_{mask_col}"

    mod_times = dw.data[mod_col].astype("datetime64[ns]")
    orig_hex_dtype = dw.data[dw.hex].dtype
    if dw.is_dask:
        dw.data[mod_col] = dd.to_datetime(mod_times, utc=True)
    else:
        dw.data[mod_col] = pd.to_datetime(mod_times, utc=True)

    # Needed because in datetime conversion, hex_id gets float-ified, somehow
    dw.data[dw.hex] = dw.data[dw.hex].astype(orig_hex_dtype)

    logger.info("Dropping coalesced dwells")
    dw.data[mask_col] = dw.data[mask_col].astype("boolean")
    dw.data[mask_col] = dw.data[mask_col].replace(False, pd.NA)
    dw.data = dw.data.dropna(subset=mask_col)
    dw.data[mask_col] = dw.data[mask_col].astype(bool)
    drop_cols = [mask_col] + accum_cols + [reset_orig_col, dw.reset]
    dw.data = dw.data.drop(columns=drop_cols)
    renamer = {f"{old}_{mask_col}": old for old in accum_cols}
    dw.data = dw.data.rename(columns=renamer)

    # Set reset back to its original column name and data type
    renamer = {reset_orig_col: dw.reset}
    dw.data = dw.data.rename(columns=renamer)
    dw.data[dw.reset] = dw.data[dw.reset].astype(bool)
    return dw


def mark_vehicle_shifts(dw: DwellSet, params: dict) -> DwellSet:
    """Identify driver shift boundaries and assign shift IDs.

    A dwell whose duration meets or exceeds ``min_refresh_hrs`` is classified
    as a shift-refresh dwell (the driver's mandatory off-duty rest period under
    FMCSA 49 CFR Part 395).  The shift ID for the *following* dwell is
    incremented, so that the refresh dwell itself is the last event of the
    preceding shift.  Shift IDs are vehicle-local (i.e., they restart at 0
    for each vehicle).

    Args:
        dw: Coalesced ``DwellSet`` (output of ``coalesce_interrupted_dwells``).
        params: Pipeline parameters dict with keys:

            - ``columns`` (dict): sub-keys:

                - ``dur`` (str): temporary column name for dwell duration.
                - ``refresh`` (str): boolean column name marking shift-refresh
                  dwells.
                - ``shift_id`` (str): integer shift-ID column name.
            - ``min_refresh_hrs`` (float): minimum dwell duration in hours to
              classify as a shift refresh (typically 6.9 hours).

    Returns:
        The ``DwellSet`` with ``refresh`` and ``shift_id`` columns added to
        ``dw.data``.
    """
    # Mark the shift "refresh" dwells
    pcols = params["columns"]
    dw.data[pcols["dur"]] = total_hours(dw.data[dw.end] - dw.data[dw.start])
    dw.data[pcols["refresh"]] = dw.data[pcols["dur"]] >= params["min_refresh_hrs"]
    dw.data = dw.data.drop(columns=[pcols["dur"]])

    # Give each shift an id number, which is unique within the vehicle, but not between
    scol = pcols["shift_id"]

    dw.data[scol] = dw.data.groupby(dw.veh)[pcols["refresh"]].cumsum()

    kws = {"fill_value": 0}
    if dw.is_dask:
        kws.update({"meta": ("x", "i8")})

    dw.data[scol] = dw.data.groupby(dw.veh)[scol].shift(1, **kws)
    return dw


def calc_rolling_dwell_ratios(dw: DwellSet, params: dict) -> DwellSet:
    """Compute each dwell's maximum rolling location-specific dwell ratio.

    For each dwell, the rolling ratio is the fraction of the vehicle's total
    dwell time (within a rolling time window) that was spent at the same
    hexagon.  The *maximum* of this ratio across all windows in the observation
    period is recorded for each (vehicle, location) pair and then mapped back
    onto individual dwell rows.

    Dispatches to ``_calc_rolling_dwell_ratios_part`` via ``map_partitions``
    for Dask-backed ``DwellSet`` instances, or calls it directly for pandas.

    Args:
        dw: Shift-annotated ``DwellSet`` (output of ``mark_vehicle_shifts``).
        params: Pipeline parameters dict with keys:

            - ``output_ratio_col`` (str): name of the output rolling-ratio
              column.
            - ``rolling_kwargs`` (dict): keyword arguments forwarded to
              ``pd.Series.rolling`` (e.g., ``window``, ``min_periods``).

    Returns:
        The ``DwellSet`` with ``params["output_ratio_col"]`` added to
        ``dw.data``.
    """
    dw.data["dur_hrs_col"] = total_hours(dw.data[dw.end] - dw.data[dw.start])

    out_col = params["output_ratio_col"]

    kws = {
        "val_col": "dur_hrs_col",
        "veh_col": dw.veh,
        "loc_col": dw.hex,
        "time_col": dw.start,
        "out_col": out_col,
    }
    kws.update(params["rolling_kwargs"])
    if dw.is_dask:
        meta = dd.utils.make_meta(dw.data)
        meta[out_col] = np.array([], dtype=np.float64)
        kws["meta"] = meta
        dw.data = dw.data.map_partitions(func=_calc_rolling_dwell_ratios_part, **kws)
    else:
        dw.data = _calc_rolling_dwell_ratios_part(dw.data, **kws)

    dw.data = dw.data.drop(columns=["dur_hrs_col"])
    return dw


def _calc_rolling_dwell_ratios_part(
    part: pd.DataFrame,
    out_col: str,
    val_col: str,
    veh_col: str,
    loc_col: str,
    time_col: str,
    **roll_kwargs: dict,
) -> pd.DataFrame:
    """Apply ``_calc_rolling_dwell_ratios_one_veh`` to each vehicle in a partition."""
    part[out_col] = part.groupby(veh_col, sort=False, group_keys=False).apply(
        func=_calc_rolling_dwell_ratios_one_veh,
        val_col=val_col,
        loc_col=loc_col,
        time_col=time_col,
        **roll_kwargs,
    )
    return part


def _calc_rolling_dwell_ratios_one_veh(
    dwells: pd.DataFrame, val_col: str, loc_col: str, time_col: str, **roll_kwargs: dict
) -> pd.Series:
    """Compute per-dwell maximum rolling location dwell ratio for a single vehicle.

    Sets ``time_col`` as the index to enable time-based rolling windows, then
    computes both the location-specific rolling sum and the overall rolling sum
    of dwell duration.  The ratio of these two quantities is computed at each
    timestamp; the *maximum* ratio observed for each location is then mapped
    back onto each dwell row.

    Args:
        dwells: Single-vehicle dwell DataFrame, sorted by ``time_col``.
        val_col: Dwell-duration column name.
        loc_col: Hexagon/location column name.
        time_col: Timestamp column name (used as the rolling index).
        **roll_kwargs: Keyword arguments forwarded to ``pd.Series.rolling``
            (e.g., ``window="30D"``).

    Returns:
        A ``pd.Series`` aligned with ``dwells`` containing the maximum rolling
        dwell ratio for the location visited at each dwell.
    """
    roller = dwells.loc[:, [time_col, loc_col, val_col]]
    roller = roller.set_index(time_col)
    loc_specific = roller.groupby(loc_col)[val_col].rolling(**roll_kwargs).sum()
    loc_specific.name = "loc_specific"
    loc_specific = loc_specific.reset_index(loc_col)
    overall = roller[val_col].rolling(**roll_kwargs).sum()
    overall.name = "overall"

    ratior = pd.concat([loc_specific, overall], axis=1, ignore_index=False)
    ratior["ratio"] = ratior["loc_specific"] / ratior["overall"]

    loc_max_ratios = ratior.groupby(loc_col)["ratio"].max()
    out = dwells[loc_col].map(loc_max_ratios)

    return out


def map_location_groups(
    dw: DwellSet, hex_corresp: pd.DataFrame, params: dict
) -> DwellSet:
    """Join freight-activity-class labels from the hex correspondence table onto dwells.

    Maps the cluster/group label for each hexagon (computed by the
    ``describe_locations`` pipeline) onto the ``DwellSet`` rows by hexagon ID.
    For Dask-backed ``DwellSet`` instances, the mapping is performed lazily
    via ``dw.data[dw.hex].map(map_ser, meta=...)``.

    Args:
        dw: ``DwellSet`` with a hex-ID column.
        hex_corresp: DataFrame indexed by hexagon ID with a location-group
            column.
        params: Pipeline parameters dict with keys:

            - ``location_group_col`` (str): name of the group column in
              ``hex_corresp`` and the output column added to ``dw.data``.
            - ``missing_values`` (dict): sub-keys:

                - ``fill_missing`` (bool): whether to fill NaN group labels.
                - ``fill_value``: value used to fill unmapped hexagons (e.g.,
                  ``"undeveloped"``).

    Returns:
        The ``DwellSet`` with a new location-group column added to ``dw.data``.
    """
    grp_col = params["location_group_col"]
    map_ser = hex_corresp[grp_col]

    if dw.is_dask:
        meta = (grp_col, map_ser.dtype)
        dw.data[grp_col] = dw.data[dw.hex].map(map_ser, meta=meta)
    else:
        dw.data[grp_col] = dw.data[dw.hex].map(map_ser)

    mpars = params["missing_values"]
    if mpars["fill_missing"]:
        dw.data[grp_col] = dw.data[grp_col].fillna(mpars["fill_value"])
    return dw


### vvv Functions below here not used as of 10/20/2025, but preserved for later use vvv


def calc_inter_visit_stats(dw: DwellSet) -> DwellSet:
    """Compute inter-visit time and mileage for each (vehicle, location) pair.

    .. deprecated::
        Not connected to the active pipeline as of 2025-10-20.  Preserved for
        potential future use.
    """
    tqdm.pandas()
    # TODO: Consider moving this within DwellSet class and using a Numba for loop
    dw.data = dw.data.groupby(dw.veh, group_keys=False, sort=False).progress_apply(
        calc_inter_visit_times, hex_col=dw.hex, end_col=dw.end, start_col=dw.start
    )

    dw.data["cum_veh_miles"] = dw.data.groupby(dw.veh, sort=False)[
        dw.trip_dist
    ].cumsum()
    dw.data["inter_visit_miles"] = dw.data.groupby([dw.veh, dw.hex], sort=False)[
        "cum_veh_miles"
    ].diff()
    dw.data = dw.data.drop(columns=["cum_veh_miles"])
    return dw


def calc_inter_visit_times(
    grp: pd.DataFrame, hex_col: str, end_col: str, start_col: str
) -> pd.DataFrame:
    """Calculate inter-visit times for a single vehicle's dwells, sorted by time."""
    prev_end_time = grp.groupby(hex_col, sort=False)[end_col].shift(1)
    grp.loc[:, "inter_visit_hrs"] = total_hours(grp[start_col] - prev_end_time)
    return grp


def describe_veh_loc_pairs(dw: DwellSet) -> pd.DataFrame:
    """Summarise each (vehicle, location) pair with visit counts, dwell hours, and inter-visit statistics.

    .. deprecated::
        Not connected to the active pipeline as of 2025-10-20.  Preserved for
        potential future use.
    """
    veh_locs = dw.data.groupby([dw.veh, dw.hex], sort=False).agg(
        n_visits=pd.NamedAgg("dwell_hrs", "count"),
        mean_inter_miles=pd.NamedAgg("inter_visit_miles", "mean"),
        med_inter_miles=pd.NamedAgg("inter_visit_miles", "median"),
        max_inter_miles=pd.NamedAgg("inter_visit_miles", "max"),
        mean_inter_times=pd.NamedAgg("inter_visit_hrs", "mean"),
        med_inter_times=pd.NamedAgg("inter_visit_hrs", "median"),
        max_inter_times=pd.NamedAgg("inter_visit_hrs", "max"),
        med_dwell_hrs=pd.NamedAgg("dwell_hrs", "median"),
        tot_dwell_hrs=pd.NamedAgg("dwell_hrs", "sum"),
        max_dwell_hrs_roll_ratio=pd.NamedAgg("dwell_hrs_roll_ratio", "max"),
    )
    veh_locs["dwell_hrs_ratio"] = veh_locs.groupby(dw.veh, sort=False)[
        "tot_dwell_hrs"
    ].transform(lambda s: s / s.sum())
    veh_locs["visits_ratio"] = veh_locs.groupby(dw.veh, sort=False)[
        "n_visits"
    ].transform(lambda s: s / s.sum())
    return veh_locs


def filter_substantial_dwells(dw: DwellSet, params: dict) -> DwellSet:
    """Drop dwells shorter than a duration threshold, accumulating trip stats across gaps.

    .. deprecated::
        Not connected to the active pipeline as of 2025-10-20.  Preserved for
        potential future use.
    """
    dw.data["dwell_hrs"] = total_hours(dw.data[dw.end] - dw.data[dw.start])
    dw.data["long_enough"] = dw.data["dwell_hrs"] > params["thresh_hrs"]
    accum_cols = [dw.trip_dist, dw.trip_dur, dw.reset]
    dw.accum_masked("long_enough", accum_cols=accum_cols, inplace=True)
    dw.data = dw.data.drop(columns=accum_cols)
    dw.data = dw.data.rename(columns={f"{col}_long_enough": col for col in accum_cols})
    dw.drop_masked(keep_mask_col="long_enough", inplace=True)
    return dw


def cluster_veh_loc_pairs(veh_locs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Cluster (vehicle, location) pairs using HDBSCAN on log-scaled, standardised features.

    .. deprecated::
        Not connected to the active pipeline as of 2025-10-20.  Preserved for
        potential future use.

    Selects ``params["feature_cols"]``, optionally subsamples, applies
    ``log10`` transformation (skipping ratio columns), standardises with
    ``StandardScaler``, and fits HDBSCAN with a minimum cluster size of
    ``n_obs / min_cluster_size_denom``.

    Args:
        veh_locs: (vehicle, location) summary DataFrame (output of
            ``describe_veh_loc_pairs``).
        params: Pipeline parameters dict with keys:

            - ``feature_cols`` (list[str]): feature columns for clustering.
            - ``sample`` (dict): ``active`` (bool), ``n`` (int), ``seed``
              (int) for optional subsampling.
            - ``min_cluster_size_denom`` (int): denominator for deriving
              HDBSCAN's ``min_cluster_size`` from ``n_obs``.
            - ``cluster_col`` (str): output cluster-label column name.

    Returns:
        The input DataFrame with a categorical ``cluster_col`` column added.
    """
    # Prepare for clustering by standardizing variables
    logger.info("Select feature variables")
    clusterable = deepcopy(veh_locs.dropna(axis=0))
    drop_cols = np.setdiff1d(clusterable.columns, params["feature_cols"])
    clusterable = clusterable.drop(columns=drop_cols)

    spars = params["sample"]
    if spars["active"]:
        n = spars["n"]
        logger.info(f"Sample {n} observations")
        clusterable = clusterable.sample(n=n, random_state=spars["seed"])

    logger.info("Log-transform and mean-std scale features.")
    for col in clusterable.columns:
        if not col.endswith("_ratio"):
            # Ratio columns would not benefit from spread reduction of log1p
            clusterable.loc[:, col] = clusterable[col] + 1
        if not col.endswith("_entropy"):
            clusterable.loc[:, col] = np.log10(clusterable[col])
    scaler = StandardScaler()
    clusterable = pd.DataFrame(
        data=scaler.fit_transform(clusterable),
        index=clusterable.index,
        columns=clusterable.columns,
    )

    # Perform clustering
    n_obs = len(clusterable)
    logger.info(f"Beginning clustering on {n_obs} observations")
    min_clust_size = int(n_obs / params["min_cluster_size_denom"])
    clusterer = HDBSCAN(min_cluster_size=min_clust_size)
    clusterer = clusterer.fit(clusterable.values)

    # Merge results back on to original dataframe
    clusters = pd.DataFrame(
        data=clusterer.labels_,
        index=clusterable.index,
        columns=[params["cluster_col"]],
    )
    clusters[params["cluster_col"]] = pd.Categorical(clusters[params["cluster_col"]])
    veh_locs = veh_locs.merge(clusters, left_index=True, right_index=True)
    return veh_locs
