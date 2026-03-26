"""Kedro pipeline nodes for the ``electrify_trips`` pipeline (Model Module 4).

This module implements the per-vehicle charging-choice simulation described in
**Module 4 ("Simulate electrified dwells")** of Passow & Rajagopal (2026).
Starting from a dataset of observed diesel-HDT dwell events, it produces a
dataset of *electrified* dwells annotated with charging decisions — which mode
was used, how much energy was transferred, and how much delay was incurred —
for a single scenario (state of the world).

Pipeline overview
-----------------
The nodes in this module are executed in the following logical order by the
Kedro pipeline:

1. **filter_vehicles** — restrict the dwell data to the vehicle cohort
   selected for the current scenario.
2. **calc_vehicle_ranges** — assign each vehicle a design range (miles) and
   battery capacity (kWh) derived from its observed shift mileage
   distribution.
3. **calc_dwell_durations** — shrink each dwell window by plug-in/plug-out
   overhead and compute net available charging time.
4. **prepare_modes** / **assign_modes** / **build_mode_power_lut** — annotate
   each dwell with the charging modes available at that location and the
   maximum deliverable power.
5. **merge_dwellset_node** — join per-vehicle parameters (e.g. consumption
   rate) into the dwell table.
6. **calc_energy_use** — compute per-trip energy demand (kWh).
7. **mark_critical_days** — classify vehicle-days where en-route charging is
   necessary because total shift energy exceeds battery capacity.
8. **filter_dwells** — drop en-route dwells that are too short or on
   non-critical days (pre-simulation filter).
9. **mark_shift_powers** — record the maximum power available later in the
   shift at each dwell, used as a look-ahead input to the charging algorithm.
10. **simulate_charging_choice** — run the forward-looking utility-
    maximisation charging-choice algorithm (Numba JIT-compiled) for every
    vehicle.
11. **filter_dwells_post** — drop optional stops where the simulation
    determined no charging occurred (post-simulation filter).

Key design decisions
--------------------
- **Critical-day filtering**: En-route (truck-stop) charging is only
  considered on vehicle-days where the total shift energy demand exceeds
  battery capacity.  This reduces the search space and avoids over-estimating
  public-charging demand on days when depot charging alone suffices.
- **Optional stops**: Proxy dwells inserted at truck stops along routed paths
  have zero net duration.  They are carried through the pre-simulation filter
  and dropped post-simulation if the vehicle chose not to charge there.
- **Dask support**: All nodes handle both pandas- and Dask-backed
  ``DwellSet`` objects.  Dask execution partitions the vehicle fleet so that
  each partition is processed independently; callers must ensure vehicle
  sequences are not split across partitions.
- **Numba JIT compilation**: The inner charging loop in
  ``simulate_charging_choice`` is JIT-compiled.  An optional pre-compilation
  step warms up the compiled functions before the main run.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform
proactive substation upgrades for charging electric heavy-duty trucks.
*Applied Energy* (submitted March 2026).

Liu, J., et al. Utility-maximisation charging choice model (inspiration for
``ForwardLookingChargingChoiceStrategy``).
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
    """Filter the dwell data to only include vehicles present in the vehicles table.

    Retains only dwell rows whose vehicle index appears in ``vehs``. Any
    vehicle in ``vehs`` that has no dwell rows generates a warning. For
    Dask-backed DwellSets the filtered result is repartitioned to a fixed
    number of partitions to rebalance work after dropping rows.

    Args:
        dw: The dwell dataset to filter. Modified in-place and returned.
        vehs: Vehicle-level table whose index contains the set of vehicle IDs
            to keep. Rows in ``dw`` whose vehicle ID is absent from this index
            are dropped.
        params: Pipeline parameters. Expected keys:

            - ``n_partitions`` (int): Target partition count used when
              repartitioning a Dask-backed ``dw`` after filtering.

    Returns:
        The filtered ``DwellSet`` containing only dwells for vehicles found in
        ``vehs``.
    """
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


def calc_vehicle_ranges(vehs: pd.DataFrame, dw: DwellSet, params: dict) -> pd.DataFrame:
    """Assign a design range (miles) and battery capacity (kWh) to each vehicle.

    Design range is determined by taking the maximum of two per-vehicle
    criteria computed from observed shift mileage distributions:

    1. **Death range** — the ``no_death_shift_frac`` quantile of each
       vehicle's single-shift longest-trip distances.  The vehicle must be
       able to complete any shift's longest leg without running out of charge.
    2. **Charge range** — the ``no_charge_shift_frac`` quantile of each
       vehicle's total-shift mileage, divided by the usable SoC band
       (``soc_buffer_high`` − ``soc_buffer_low``).  The vehicle should be
       able to complete a typical full shift on one charge.

    The continuous desired range is then rounded *up* to the nearest value in
    ``range_options_miles`` (via ``pd.cut`` with the top bin open), and
    multiplied by the vehicle's energy-consumption rate to obtain battery
    capacity.

    Args:
        vehs: Vehicle-level table to augment.  Two columns are added in-place:
            the design range column and the battery capacity column (names
            taken from ``params``).  The vehicle index is used to join shift
            statistics back to vehicles.
        dw: Dwell dataset containing trip-distance and shift-ID columns used
            to derive per-vehicle shift mileage statistics.
        params: Pipeline parameters. Expected keys:

            - ``columns`` (dict): Column-name mappings with sub-keys:

              - ``shift`` — shift identifier column in ``dw.data``
              - ``range_mi`` — output design-range column name in ``vehs``
              - ``batt_kwh`` — output battery-capacity column name in ``vehs``
              - ``consump_kwh_per_mi`` — energy-consumption-rate column in
                ``vehs``

            - ``no_death_shift_frac`` (float): Quantile (0–1) of the
              longest-trip distribution used for the death-range criterion.
            - ``no_charge_shift_frac`` (float): Quantile (0–1) of the
              total-shift-miles distribution used for the charge-range
              criterion.
            - ``soc_buffer_high`` (float): Upper SoC target (fraction, 0–1).
            - ``soc_buffer_low`` (float): Lower SoC buffer (fraction, 0–1).
            - ``range_options_miles`` (list[float]): Ordered list of
              candidate design ranges in miles (e.g. ``[150, 300, 500]``).
              The top entry acts as the ceiling; everything above the
              second-to-last bin edge is mapped to the top option.

    Returns:
        The ``vehs`` DataFrame with two new columns appended: design range
        (miles) and battery capacity (kWh).
    """
    pcols = params["columns"]

    # Describe shifts
    shift_miles = dw.data.groupby([dw.veh, pcols["shift"]])[dw.trip_dist].agg(
        ["sum", "max"]
    )
    renamer = {"sum": "shift_total_miles", "max": "shift_longest_trip"}
    shift_miles = shift_miles.rename(columns=renamer)
    if dw.is_dask:
        shift_miles = shift_miles.compute()
    shift_miles = shift_miles.reset_index()

    # Compute desired ranges for each criterion
    veh_grp = shift_miles.groupby(dw.veh)
    kws = {"interpolation": "linear"}

    q_dth = params["no_death_shift_frac"]
    death_range = veh_grp["shift_longest_trip"].quantile(q=q_dth, **kws)

    q_chg = params["no_charge_shift_frac"]
    charge_range = veh_grp["shift_total_miles"].quantile(q=q_chg, **kws)
    desired_soc_band = params["soc_buffer_high"] - params["soc_buffer_low"]
    charge_range = charge_range / desired_soc_band

    # Compute design range from desired ranges for each criterion
    ranges = pd.concat([death_range, charge_range], axis=1)
    ranges["range_desired"] = ranges.max(axis=1)

    # Omitting top bin to automatically do the two-way rounding
    ropts = params["range_options_miles"]
    range_bins = [0.0] + ropts[:-1] + [np.inf]
    veh_bins = pd.cut(ranges["range_desired"], bins=range_bins, labels=ropts)

    # Assign ranges to vehicles
    vehs[pcols["range_mi"]] = vehs.index.map(veh_bins).astype(float)

    # Assign battery capacities to vehicles matching ranges
    vehs[pcols["batt_kwh"]] = (
        vehs[pcols["range_mi"]] * vehs[pcols["consump_kwh_per_mi"]]
    )

    return vehs


def calc_dwell_durations(dw: DwellSet, params: dict) -> DwellSet:
    """Compute net charging-available dwell duration for each stop.

    Converts raw plug-in and plug-out overhead times from a numeric unit to
    ``timedelta``, then shrinks each dwell's start and end times inward by
    those overhead amounts.  The remaining window represents the time actually
    available for charging.  Optional stops — identified by identical start
    and end times — are left with zero duration so they are not double-counted.

    The resulting net dwell duration (in hours, as a float) is written to a
    new column.

    Args:
        dw: Dwell dataset. ``dw.start`` and ``dw.end`` are timestamp columns
            that bound each dwell. Modified in-place and returned.
        params: Pipeline parameters. Expected keys:

            - ``in_out_time_cols`` (dict): Maps ``"plug_in"`` and
              ``"plug_out"`` to the column names in ``dw.data`` that hold the
              overhead durations (in the unit given by ``in_out_time_unit``).
            - ``in_out_time_unit`` (str): Time unit string recognised by
              ``pd.to_timedelta`` / ``dd.to_timedelta`` (e.g. ``"s"`` for
              seconds, ``"min"`` for minutes).
            - ``dwell_time_col`` (str): Name of the output column that will
              hold the net dwell duration in hours.

    Returns:
        The updated ``DwellSet`` with adjusted ``dw.start`` / ``dw.end``
        timestamps and a new float column for net dwell hours.
    """
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
    """Convert the charging-modes parameter dict into a tidy DataFrame.

    The ``modes`` parameter dict maps mode names to their attribute dicts
    (e.g. maximum power).  Two special keys — ``name_column`` and
    ``id_column`` — control the column and index names of the resulting table
    and are removed before conversion.

    Example input (YAML)::

        modes:
          name_column: mode_name
          id_column: mode_id
          depot:
            max_power_kw: 150
          truck_stop:
            max_power_kw: 350

    Produces a DataFrame with index name ``mode_id``, a ``mode_name`` column
    (``"depot"``, ``"truck_stop"``, …), and one column per attribute.

    Args:
        modes: Charging-modes parameter dictionary, typically loaded from the
            Kedro parameters YAML.  Must contain the two special keys
            ``name_column`` and ``id_column``; all remaining keys are treated
            as mode names whose values are attribute dicts.

    Returns:
        A ``pd.DataFrame`` with one row per charging mode, indexed by a
        sequential integer ID, and columns for the mode name and each
        mode attribute.
    """
    modes_copy = dict(modes)  # avoid mutating input params dict
    name_col = modes_copy.pop("name_column")
    id_col = modes_copy.pop("id_column")
    modes_df = pd.DataFrame.from_dict(data=modes_copy, orient="index")
    modes_df.index.name = name_col
    modes_df = modes_df.reset_index()
    modes_df.index.name = id_col
    return modes_df


def assign_modes(dw: DwellSet, modes: pd.DataFrame, params: dict) -> DwellSet:  # noqa: PLR0912
    """Annotate each dwell with available charging modes and maximum deliverable power.

    Charging-mode availability at a dwell is determined by two independent
    mechanisms and then encoded into a compact integer bitmask:

    1. **Location-based modes** (e.g. truck-stop charging): A boolean column
       is created for each mode whose availability depends on the dwell's
       location group.  The selector can be a literal ``True``/``False`` or a
       dict specifying which ``loc_groups`` values enable the mode (with an
       optional ``invert_selection`` flag).

    2. **Vehicle-based depot mode**: A single mode whose availability depends
       on whether the vehicle's home-base ratio exceeds a threshold.  The
       ratio column is supplied via ``params["veh_based_mode_avail"]``.

    After building one boolean column per mode, all boolean columns are
    combined into an integer bitmask (via ``bool_arr_to_bits``), and the
    maximum power deliverable under each possible bitmask combination is
    looked up from a pre-built LUT (see ``build_mode_power_lut``).  The
    individual boolean mode columns are then dropped.

    Args:
        dw: Dwell dataset to annotate. Modified in-place and returned.
        modes: Modes table produced by ``prepare_modes``, with one row per
            charging mode and columns for mode name and maximum power.
        params: Pipeline parameters. Expected keys:

            - ``mode_col`` (str): Column in ``modes`` holding mode names.
            - ``loc_based_mode_avail`` (dict): Maps mode name → selector.
              Each selector is either a ``bool`` (applies globally) or a dict
              with ``loc_groups`` (list of location-group values that enable
              the mode) and optionally ``invert_selection`` (bool).
            - ``loc_group_col`` (str): Column in ``dw.data`` containing the
              location-group identifier for each dwell.
            - ``veh_based_mode_avail`` (dict): Configuration for the one
              vehicle-home-based mode, with sub-keys ``mode_name`` (str),
              ``ratio_col`` (str), and ``ratio_thresh`` (float).
            - ``mode_mask_col`` (str): Output column name for the integer
              mode-availability bitmask.
            - ``max_power_source_col`` (str): Column in ``modes`` holding
              each mode's maximum power (kW).
            - ``max_power_col`` (str): Output column name for the maximum
              deliverable power at each dwell.

    Returns:
        The annotated ``DwellSet`` with a mode-availability bitmask column
        and a maximum-power column added; individual per-mode boolean columns
        are removed.

    Raises:
        ValueError: If any mode name in ``modes`` already exists as a column
            in ``dw.data``, if a mode specified in ``loc_based_mode_avail``
            is not present in ``modes``, if a selector value is neither
            ``bool`` nor ``dict``, or if any mode remains unassigned after
            processing both location- and vehicle-based rules.
    """
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
    """Pre-compute the maximum deliverable power for every mode-availability bitmask.

    Enumerates all 2^N combinations of N charging modes (where N is
    ``len(mode_names)``), encodes each combination as an integer bitmask via
    ``bool_arr_to_bits``, and records the maximum mode power across all
    *enabled* modes in that combination.  A bitmask of zero (no modes
    available) maps to 0.0 kW.

    This lookup table is used by ``assign_modes`` so that per-dwell maximum
    power can be obtained by a single O(1) dictionary lookup rather than
    recomputing the max over enabled modes for every row.

    Args:
        mode_names: Ordered series of charging-mode name strings.  The order
            determines bit positions in the bitmask (first name → least
            significant bit).
        mode_powers: Series of maximum power values (kW), aligned with
            ``mode_names`` by position.

    Returns:
        A ``dict`` mapping each integer bitmask (``int``) to the maximum
        deliverable power (``float``, kW) across all modes enabled in that
        bitmask.
    """
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
    """Merge a DataFrame into a DwellSet using the standard merge-node helper.

    A thin wrapper around ``merge_dataframes_node`` that extracts the
    underlying DataFrame from ``dw``, performs the merge, and writes the
    result back.  All merge semantics (join type, key columns, etc.) are
    controlled by ``params`` exactly as they would be for a plain DataFrame
    merge node.

    Args:
        dw: Dwell dataset whose ``data`` attribute is used as the left side
            of the merge. Modified in-place and returned.
        right: DataFrame to merge in on the right side.
        params: Merge parameters forwarded verbatim to
            ``merge_dataframes_node``.  Refer to that helper for the full
            parameter schema.

    Returns:
        The updated ``DwellSet`` with ``dw.data`` replaced by the merged
        result.
    """
    dw.data = merge_dataframes_node(
        left=dw.data,
        right=right,
        params=params,
    )
    return dw


def calc_energy_use(dw: DwellSet, params: dict) -> DwellSet:
    """Compute per-trip energy demand (kWh) as trip distance × consumption rate.

    Multiplies the trip-distance column (``dw.trip_dist``) by a
    vehicle-specific energy-consumption rate column to produce per-trip energy
    demand in kWh.  Both columns must already be present in ``dw.data``; the
    result is written to a new column.

    Args:
        dw: Dwell dataset containing trip distances and energy-consumption
            rates. Modified in-place and returned.
        params: Pipeline parameters. Expected keys:

            - ``energy_col`` (str): Name for the output energy column (kWh).
            - ``consump_col`` (str): Name of the column in ``dw.data`` holding
              each vehicle's energy consumption rate (kWh/mile).

    Returns:
        The updated ``DwellSet`` with the new energy column appended.
    """
    dw.data[params["energy_col"]] = (
        dw.data[dw.trip_dist] * dw.data[params["consump_col"]]
    )
    return dw


def mark_critical_days(dw: DwellSet, params: dict) -> DwellSet:
    """Classify each dwell as belonging to a critical or non-critical vehicle-day.

    A vehicle-day is **critical** when the total energy demand for all
    remaining trips in the shift exceeds the vehicle's battery capacity,
    making en-route charging necessary.  Non-critical days can
    be completed on a single depot/destination charge, so en-route dwells on
    those days are candidates for filtering.

    The algorithm proceeds in three steps:

    1. **Identify refresh boundaries** — a refresh dwell is one that has
       sufficient dwell duration to fully recharge the battery (duration ≥
       battery capacity / max power).  These mark the segment boundaries
       within which energy demand is accumulated.

    2. **Accumulate remaining-shift energy** — starting from each refresh
       boundary and working backwards, the energy demand of subsequent trips
       is summed up to the next refresh boundary.  This gives the energy a
       vehicle would need on-board if it arrived at the refresh stop empty.

    3. **Classify and propagate** — a dwell is initially critical if its
       accumulated remaining-shift energy exceeds battery capacity.  The
       critical flag is then forward-filled within each vehicle's sequence so
       that *all* dwells between a refresh boundary and the critical trip
       inherit the flag.  Partial days (no prior refresh boundary) are
       conservatively treated as critical.

    Args:
        dw: Dwell dataset with energy, duration, power, battery-capacity, and
            refresh columns already populated. Modified in-place and returned.
        params: Pipeline parameters. Expected keys:

            - ``refresh_col`` (str): Boolean column marking refresh-eligible
              dwells (e.g. truck-stop dwells with sufficient dwell time).
            - ``crit_bound_col`` (str): Temporary column name used internally
              for the refresh-and-can-charge boundary flag.
            - ``batt_cap_col`` (str): Column holding each dwell row's vehicle
              battery capacity (kWh).
            - ``max_power_col`` (str): Column holding maximum available
              charging power at each dwell (kW).
            - ``dur_col`` (str): Column holding net dwell duration (hours).
            - ``energy_col`` (str): Column holding per-trip energy demand
              (kWh).
            - ``energy_col_next_trip`` (str): Temporary column for the next
              trip's energy demand.
            - ``energy_col_remain_shift`` (str): Output column for total
              remaining-shift energy demand (kWh).
            - ``crit_col`` (str): Output boolean column marking critical-day
              dwells.

    Returns:
        The updated ``DwellSet`` with a new boolean ``crit_col`` column and
        the intermediate ``crit_bound_col`` column removed.
    """
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
    """Drop unwanted dwell rows and propagate corrected cumulative columns.

    This is the shared implementation called by both ``filter_dwells`` and
    ``filter_dwells_post``.  It wraps ``DwellSet.accum_masked()`` with the
    bookkeeping needed to:

    - Accumulate trip-distance, trip-duration, and reset columns (and any
      extra columns) through the mask so that the values on *kept* rows
      correctly reflect the skipped rows.
    - Drop rows where ``keep_col`` is ``False``.
    - Rename the accumulated ``<col>_<keep_col>`` output columns back to their
      original names so downstream nodes are unaffected.

    The base set of forward-accumulated columns is always
    ``[dw.trip_dist, dw.trip_dur, dw.reset]``.  Callers may supply additional
    forward (``accum_cols_fw_extra``) and reverse (``accum_cols_rv``)
    accumulation columns.

    Args:
        dw: Dwell dataset to filter. Modified in-place and returned.
        keep_col: Name of the boolean column in ``dw.data`` indicating which
            rows to keep (``True``) or drop (``False``).
        accum_cols_fw_extra: Additional column names to accumulate in the
            forward direction (i.e. values propagate from preceding kept rows
            to the next kept row). ``None`` means no extras.
        accum_cols_rv: Column names to accumulate in the *reverse* direction
            (i.e. values propagate from following kept rows back to the
            current row). ``None`` means no reverse accumulation.
        drop_cols_extra: Additional column names to drop after filtering,
            beyond ``keep_col`` and the original (pre-rename) accumulated
            columns. ``None`` means no extras.

    Returns:
        The filtered and updated ``DwellSet`` with corrected cumulative
        columns and dropped rows.
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
    """Drop en-route dwells that cannot meaningfully contribute to charging.

    Applied *before* the charging-choice simulation. A dwell is dropped when:

    - Its net available duration is negative (shorter than plug-in +
      plug-out overhead), OR
    - (When ``filter_critical_days`` is enabled) it is neither a refresh
      location nor part of a critical day, AND it is not an optional stop.

    Keeping optional stops (zero-duration proxy stops inserted along routes)
    regardless of the critical-day flag preserves them for the
    post-simulation filter, which uses actual charging decisions to decide
    whether they were visited.

    Dropped rows are handled via ``_filter_dwells_core``, which accumulates
    trip-distance and related columns through the mask before removing rows
    so that successive kept rows maintain correct running totals.

    Args:
        dw: Dwell dataset to filter. Modified in-place and returned.
        params: Pipeline parameters. Expected keys:

            - ``dwell_time_col`` (str): Column holding net dwell duration in
              hours (negative means too short to plug in/out).
            - ``filter_critical_days`` (bool): Whether to apply the
              critical-day filter in addition to the duration filter.
            - ``filter_cols`` (dict): Required when ``filter_critical_days``
              is ``True``.  Sub-keys:

              - ``refresh`` — boolean column marking refresh-eligible dwells.
              - ``crit`` — boolean column marking critical-day dwells.

            - ``accum_cols_forward_extra`` (list[str]): Extra columns to
              accumulate in the forward direction through the mask.
            - ``accum_cols_reverse`` (list[str]): Columns to accumulate in
              the reverse direction through the mask.
            - ``drop_cols`` (list[str]): Additional columns to drop after
              filtering.

    Returns:
        The filtered ``DwellSet`` with adjusted cumulative columns and
        reduced row count.
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
    """Record the maximum charging power available later in the shift at each dwell.

    The forward-looking charging-choice algorithm needs to know whether a
    vehicle will have access to a high-power charger later in the shift
    before it decides whether to charge now.  This node pre-computes that
    look-ahead value.

    The approach:

    1. For dwells that are *not* refresh or critical, set their available
       power to 0 (they will be auto-skipped by the charging algorithm).
    2. Reverse-accumulate the max power with a ``CumAggFunc.MAX`` within
       each refresh segment, writing the result to every dwell in the
       segment (``write_all=True``).
    3. At refresh dwells themselves, overwrite the accumulated value with a
       sentinel (``params["final_value"]``), since at a refresh stop the
       remaining power budget resets.
    4. Shift the accumulated column backward by one position so each dwell
       sees the *future* maximum, not its own value.

    Args:
        dw: Dwell dataset with refresh, critical, and max-power columns
            already populated. Modified in-place and returned.
        params: Pipeline parameters. Expected keys:

            - ``refresh_col`` (str): Boolean column marking refresh-eligible
              dwells.
            - ``crit_col`` (str): Boolean column marking critical-day dwells.
            - ``max_power_col`` (str): Column holding per-dwell maximum
              charging power (kW).
            - ``final_value`` (float): Sentinel power value assigned to
              refresh dwells after accumulation (typically the maximum
              possible charger power, kW).
            - ``fill_value`` (float): Fill value used for the shift at the
              end of a vehicle's sequence (typically 0.0).
            - ``max_power_col_shift`` (str): Output column name for the
              shifted look-ahead power (kW).

    Returns:
        The updated ``DwellSet`` with a new look-ahead power column and
        intermediate columns removed.
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
    """Run the forward-looking charging-choice simulation for all vehicles.

    For each vehicle, the ``ForwardLookingChargingChoiceStrategy`` iterates
    through dwells in chronological order and selects — at each stop — the
    charging mode and energy amount that maximises a utility function while
    respecting battery, power, and delay constraints.  The strategy is
    inspired by Liu et al. and is implemented with Numba JIT compilation for
    speed.

    Supports both pandas (single-process) and Dask (distributed) backends.
    For Dask, the simulation runs independently on each partition; it is the
    caller's responsibility to ensure that each partition contains complete,
    sorted vehicle sequences (no vehicle spans multiple partitions).

    Optionally pre-compiles the Numba JIT functions using a small mock dataset
    before the main run.  Pre-compilation is useful for single-process runs
    but does not help distributed Dask workers (each worker JIT-compiles on
    first use).

    Args:
        dw: Dwell dataset sorted by vehicle and time.  For pandas-backed
            DwellSets, sorting is performed automatically; for Dask-backed
            DwellSets, the caller must guarantee sort order is preserved
            across partitions.
        vehs: Vehicle-level table with battery capacity and other per-vehicle
            parameters required by the charging-choice strategy.
        modes: Charging-modes table produced by ``prepare_modes``, with one
            row per mode and columns for power limits and other attributes.
        params: Pipeline parameters. Expected keys:

            - ``input_cols`` (dict): Column-name mappings forwarded to
              ``ForwardLookingChargingChoiceStrategy.__init__``.  Must
              include ``modes_avail`` (bitmask column name) among others.
            - ``precompile`` (bool): Whether to trigger Numba pre-compilation
              before the main simulation run.
            - ``drop_cols`` (list[str]): Columns to drop from ``dw.data``
              after the simulation (e.g. intermediate look-ahead columns).

    Returns:
        The updated ``DwellSet`` with charging-decision columns added (charge
        amount, mode chosen, accumulated delay, etc.) and ``drop_cols``
        removed.
    """
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
    """Drop optional stops where the vehicle chose not to charge.

    Applied *after* the charging-choice simulation. Optional stops are proxy
    dwells with zero net duration that were inserted at potential en-route
    charging locations (e.g. truck stops along a routed path).  If the
    simulation determined that no charging occurs at an optional stop, that
    row carries no useful information and is removed to keep the output
    dataset compact.

    A dwell is dropped when both conditions hold:

    - It is an optional stop (``dw.end <= dw.start``), AND
    - The simulated charge amount is zero.

    Non-optional (real) dwells are always kept regardless of charge amount.

    When ``filter_unused_optionals`` is ``False``, this function is a no-op.

    Args:
        dw: Dwell dataset after charging simulation, containing charge-amount
            and timing columns. Modified in-place and returned.
        params: Pipeline parameters. Expected keys:

            - ``filter_unused_optionals`` (bool): Whether to apply the filter.
              Set to ``False`` to retain all optional stops (e.g. for
              debugging).
            - ``filter_cols`` (dict): Required when
              ``filter_unused_optionals`` is ``True``.  Sub-key:

              - ``charge`` — column holding the simulated charge amount (kWh)
                for each dwell.

            - ``accum_cols_forward_extra`` (list[str]): Extra columns to
              accumulate in the forward direction through the mask.
            - ``accum_cols_reverse`` (list[str]): Columns to accumulate in
              the reverse direction through the mask.
            - ``drop_cols`` (list[str]): Additional columns to drop after
              filtering.

    Returns:
        The filtered ``DwellSet`` with unused optional stops removed and
        cumulative columns corrected.
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
