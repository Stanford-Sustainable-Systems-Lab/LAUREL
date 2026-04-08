"""Time-series spreading and sparse quantile summarisation utilities.

Provides two core classes used throughout the load-profile assembly pipeline:

- :class:`IntervalBeginSpreader`: given a DataFrame of events with start
  timestamps and durations, "spreads" each event onto every ``freq``-aligned
  time-bin beginning that it covers.  Used by
  :func:`~laurel.models.sampling.discretize_sparse_profiles` to assign
  constant-power observations to each hourly bin they span.

- :class:`NonzeroGroupedSummarizer`: computes quantiles of sparse data where
  many possible observations are zero but only the non-zero values are
  stored.  Correctly pads with the appropriate number of zeros before
  quantile calculation.  Used to compress many-draw bootstrap profiles to
  a small set of quantiles (e.g. 20th, 50th, 80th, 95th percentile) per
  (substation, hour) cell.

Key design decisions
--------------------
- **Interval-begin convention**: the spreader produces new rows at the start of
  each covered time bin (not the end).  This is consistent with the step-
  function load-profile convention used throughout the pipeline: a constant
  power level ``p`` starts at time ``t`` and persists until the next event.
- **UTC-only timestamps**: :meth:`IntervalBeginSpreader.spread` requires the
  time column to be either timezone-naive or UTC; other timezones raise an
  error.  Timezone-naive inputs are temporarily localised to UTC for the
  spreading arithmetic and then de-localised on output.
- **``IndexIntegerizer`` for group labels**: group columns may contain string
  or categorical labels that are expensive to compare.  The spreader converts
  them to compact integer codes before passing to the JIT core and restores
  the original labels afterward.
- **Zero-padding in quantiles**: :class:`NonzeroGroupedSummarizer` only stores
  non-zero events; the ``possible_count_col`` tells the summariser how many
  total observations exist (including zeros) so that quantiles correctly
  account for the zero-inflation.  The JIT core
  (:meth:`~NonzeroGroupedSummarizer._calc_sparse_quantiles_core`) pads with
  zeros at the front of the sorted array.
"""

import datetime
from typing import Self

import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm

from laurel.utils.data import IndexIntegerizer, get_basic_dtype_ser


class IntervalBeginSpreader:
    """Expand time-interval events onto every ``freq``-aligned bin start they cover.

    For each row in the input DataFrame that crosses at least one frequency
    boundary, creates additional rows — one per covered bin beginning — with
    the same value(s) as the original row.  Rows that fit entirely within a
    single bin are kept as-is.

    This implements the "interval-begin" step-function convention: a constant
    power level ``p`` that starts at time ``t`` and lasts for ``dur`` is
    represented by one row at ``t`` and one additional row at each
    ``freq``-boundary inside ``(t, t + dur)``.

    Attributes:
        time_col: Name of the timestamp column.
        dur_col: Name of the duration column (``pd.Timedelta``-compatible).
        value_cols: Column name(s) whose values are replicated at each
            spread timestamp.
        group_cols: Column name(s) used as grouping keys; preserved in output.
        freq: Pandas frequency string defining the bin width (e.g. ``"1h"``).
    """

    time_col: str
    dur_col: str
    value_cols: list[str]
    group_cols: list[str]
    freq: str

    def __init__(
        self: Self,
        time_col: str,
        dur_col: str,
        value_cols: list[str] | str,
        group_cols: list[str] | str,
        freq: str,
    ) -> None:
        """Initialize the IntervalBeginSpreader.

        Args:
            time_col (str): Name of the column containing timestamps.
            dur_col (str): Name of the column containing event durations.
            value_cols (list[str] | str): Name(s) of column(s) containing values to be spread.
            group_cols (list[str]): Name(s) of column(s) to group by.
            freq (str): Frequency string for time block definition (e.g., '1H', '1D').
        """
        self.time_col = time_col
        self.dur_col = dur_col
        self.value_cols = [value_cols] if isinstance(value_cols, str) else value_cols
        self.group_cols = [group_cols] if isinstance(group_cols, str) else group_cols
        self.freq = freq

    def spread(
        self: Self, obs: pd.DataFrame, return_spreaded_only: bool = False
    ) -> pd.DataFrame:
        """Expand events to cover all ``freq``-aligned bin beginnings within their duration.

        Identifies rows in ``obs`` whose event end-time crosses at least one
        ``freq`` boundary (``is_overflow``).  For those rows, calls
        :meth:`_spread_wrapper` → :meth:`_spread_core` (JIT) to produce one
        output row per covered bin start.  Rows that do not overflow are kept
        unchanged (the original row already represents the bin it starts in).

        When ``return_spreaded_only=False`` (default), the original rows minus
        the ``dur_col`` column are appended to the spread rows so the output
        contains both the original event positions and the intermediate bin
        starts.

        Args:
            obs: Input DataFrame with :attr:`time_col`, :attr:`dur_col`,
                :attr:`value_cols`, and :attr:`group_cols` columns.
            return_spreaded_only: If ``True``, return only the newly generated
                intermediate bin rows (not the original event rows).  Defaults
                to ``False``.

        Returns:
            DataFrame with :attr:`group_cols`, :attr:`time_col`, and
            :attr:`value_cols` columns, with original dtypes preserved.

        Raises:
            RuntimeError: If :attr:`time_col` has a timezone other than UTC.
            ValueError: If any duration in :attr:`dur_col` is negative.
        """
        source_tz = obs[self.time_col].dt.tz
        if source_tz == datetime.UTC:
            send_to_utc = False  # If source time is UTC, then it is already compatible
        elif source_tz is None:
            send_to_utc = True
        else:
            raise RuntimeError("The source time column must be time zone naïve or UTC.")

        if np.any(obs[self.dur_col] < pd.Timedelta(0)):
            raise ValueError(
                "The duration column must have all non-negative durations."
            )

        end_of_time_group = obs[self.time_col].dt.ceil(self.freq)
        end_of_event = obs[self.time_col] + obs[self.dur_col]
        is_overflow = end_of_time_group < end_of_event
        need_spreading = obs.loc[is_overflow]

        do_spreading = len(need_spreading) > 0
        if do_spreading:
            orig_idx = need_spreading.index.names
            if orig_idx != [None]:
                need_spreading = need_spreading.reset_index()
            need_spreading = need_spreading.set_index(self.group_cols)
            group_inter = IndexIntegerizer(int_col="codes")
            need_spreading = group_inter.integerize(need_spreading)
            need_spreading = need_spreading.reset_index()
            if send_to_utc:
                need_spreading[self.time_col] = need_spreading[
                    self.time_col
                ].dt.tz_localize(datetime.UTC)

            spreaded = self._spread_wrapper(need_spreading)

            if send_to_utc:
                spreaded[self.time_col] = spreaded[self.time_col].dt.tz_localize(None)
            spreaded = spreaded.set_index("codes")
            spreaded = group_inter.deintegerize(spreaded)
            spreaded = spreaded.reset_index()
        else:
            spreaded = need_spreading
            spreaded = spreaded.drop(columns=self.dur_col)

        if not return_spreaded_only:
            keep_cols = self.group_cols + [self.time_col] + self.value_cols
            not_spread = obs.reset_index().loc[:, keep_cols]
            result = pd.concat([spreaded, not_spread], axis=0, ignore_index=True)
        else:
            result = spreaded

        return result

    def _spread_wrapper(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to NumPy arrays, call the JIT core, and reconstruct.

        Prepares inputs for :meth:`_spread_core` by:

        1. Computing floor-of-start and floor-of-end timestamps as ``int64``
           nanoseconds.
        2. Converting group codes and value columns to basic NumPy dtypes via
           :func:`~laurel.utils.data.get_basic_dtype_ser`.
        3. Calling :meth:`_spread_core` separately for each value column (so
           that each can have its own dtype in the output).
        4. Using the first column's group and time arrays (which are identical
           across columns) and combining value arrays into the final DataFrame.

        Args:
            df: Subset of overflow rows with integer group codes in the
                ``"codes"`` column and UTC-localised timestamps.

        Returns:
            DataFrame with ``"codes"``, :attr:`time_col`, and all
            :attr:`value_cols` columns at their original dtypes.
        """
        orig_time_type = df[self.time_col].dtype
        starts = df[self.time_col].dt.floor(self.freq)
        ends = (df[self.time_col] + df[self.dur_col]).dt.floor(self.freq)

        # Common arrays for all columns
        starts_int64 = starts.values.astype(np.int64)
        ends_int64 = ends.values.astype(np.int64)
        grps_array = get_basic_dtype_ser(df["codes"]).values
        tstep_ns = pd.Timedelta(self.freq).value

        # Process each value column separately
        all_results = {}
        for col in self.value_cols:
            original_dtype = df[col].dtype
            vals_array = get_basic_dtype_ser(df[col]).values

            out_grps, out_times, out_vals = self._spread_core(
                starts=starts_int64,
                ends=ends_int64,
                vals=vals_array,
                grps=grps_array,
                tstep_ns=tstep_ns,
            )

            # Convert back to original dtype
            out_vals_typed = out_vals.astype(original_dtype)

            all_results[col] = {
                "values": out_vals_typed,
                "groups": out_grps,
                "times": out_times,
            }

        # Use first column's groups and times (should be identical across columns)
        first_col = next(iter(all_results.keys()))
        out_grps = all_results[first_col]["groups"]
        out_times = all_results[first_col]["times"]
        out_times = pd.to_datetime(out_times.astype("datetime64[ns]"), utc=True)
        out_times = out_times.astype(orig_time_type)

        # Build output DataFrame
        result_dict = {
            "codes": out_grps,
            self.time_col: out_times,
        }
        for col_name, results in all_results.items():
            result_dict[col_name] = results["values"]

        result_df = pd.DataFrame(result_dict)

        return result_df

    @staticmethod
    @jit
    def _spread_core(
        starts: np.ndarray[np.int64],
        ends: np.ndarray[np.int64],
        vals: np.ndarray,
        grps: np.ndarray,
        tstep_ns: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand out events to cover all intermediate time units given their start and end
        timestamps.

        To work properly, the starts and ends arguments must be arrays of np.datetime64[h]
        which have been cast to integer.

        Args:
            starts (np.ndarray[np.int64]): Array of start timestamps as integers.
            ends (np.ndarray[np.int64]): Array of end timestamps as integers.
            vals (np.ndarray): Array of values to be expanded (single column).
            grps (np.ndarray): Array of group identifiers.
            tstep_ns (int): Time step in nanoseconds.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
                - grps_exp: Expanded group identifiers
                - times_exp: Expanded timestamps
                - vals_exp: Expanded values (single column)
        """
        n_periods = starts.shape[0]
        ends_plus = ends + tstep_ns
        starts_plus = starts + tstep_ns
        n_stamps = np.floor_divide(ends_plus - starts_plus, tstep_ns)
        tot_stamps = np.sum(n_stamps)
        times_exp = np.zeros(tot_stamps, dtype=starts_plus.dtype)
        vals_exp = np.zeros(tot_stamps, dtype=vals.dtype)
        grps_exp = np.zeros(tot_stamps, dtype=grps.dtype)
        cursor = 0
        for i in range(n_periods):
            next_cursor = cursor + n_stamps[i]
            idxs = np.arange(cursor, next_cursor, dtype=np.int64)
            times_exp[idxs] = np.arange(
                starts_plus[i], ends_plus[i], tstep_ns, dtype=times_exp.dtype
            )
            vals_exp[idxs] = vals[i]
            grps_exp[idxs] = grps[i]
            cursor = next_cursor
        return grps_exp, times_exp, vals_exp


class NonzeroGroupedSummarizer:
    """Compute quantiles of zero-inflated grouped data stored in sparse (non-zero-only) form.

    The input DataFrame contains only the non-zero values from a larger
    population.  A ``possible_count_col`` specifies the true population size
    (including zeros) for each group.  The summariser pads each group's
    non-zero values with the appropriate number of zeros before computing
    quantiles, so the result correctly reflects the full distribution.

    This is used to compress many-draw bootstrap load profiles (stored as only
    non-zero kW values) to a small set of quantile estimates per
    (substation, hour-of-week) cell.

    Attributes:
        group_cols: Column name(s) to group by.
        quantiles: Array of quantile fractions to compute (e.g.
            ``[0.2, 0.5, 0.8, 0.95]``).
        value_cols: Default value column(s); can be overridden per
            :meth:`summarize` call.
    """

    group_cols: list[str]
    quantiles: np.ndarray
    value_cols: list[str] | None

    def __init__(
        self: Self,
        group_cols: list[str],
        quantiles: np.ndarray,
        value_cols: list[str] | str | None = None,
    ) -> None:
        """Initialize the NonzeroGroupedSummarizer.

        Args:
            group_cols (list[str]): List of column names to group by.
            quantiles (np.ndarray): Array of quantile values to calculate (e.g., [0.25, 0.5, 0.75]).
            value_cols (list[str] | str | None): Name(s) of column(s) containing values to calculate
                quantiles for. Can be a single string, list of strings, or None.
                If None, value columns must be specified in summarize() method.
        """
        self.group_cols = group_cols
        self.quantiles = quantiles
        if value_cols is None:
            self.value_cols = None
        else:
            self.value_cols = (
                [value_cols] if isinstance(value_cols, str) else value_cols
            )

    def summarize(
        self: Self,
        events: pd.DataFrame,
        value_cols: list[str] | str,
        possible_count_col: str,
    ) -> pd.DataFrame:
        """Compute zero-padded quantiles per group for one or more value columns.

        For each group, retrieves the non-zero values and the total population
        count (from ``possible_count_col``), then calls the JIT core to pad
        with zeros and compute quantiles.  Processes multiple value columns in
        a loop and concatenates results column-wise.

        Args:
            events: DataFrame containing at least :attr:`group_cols`,
                ``value_cols``, and ``possible_count_col``.  Should contain
                only non-zero rows.
            value_cols: Value column(s) for which quantiles are computed.
            possible_count_col: Column giving the total count of possible
                observations (including zeros) for each group.  Must be
                constant within each group.

        Returns:
            DataFrame indexed by the grouping key(s) with columns named
            ``"{value_col}_{quantile}"`` for each (value_col, quantile)
            combination.

        Raises:
            ValueError: If the DataFrame is empty, if any group has more
                non-zero observations than its ``possible_count_col`` value,
                or if no group columns are provided.
        """
        # Convert to list format
        cols_to_process = [value_cols] if isinstance(value_cols, str) else value_cols

        # Handle empty DataFrame case
        if len(events) == 0:
            raise ValueError("Cannot process empty DataFrame.")

        grouping = events.groupby(self.group_cols, observed=True)
        grp_idxs = grouping.indices
        counts = get_basic_dtype_ser(events[possible_count_col]).values

        # Process each value column
        all_results = {}
        for col in tqdm(cols_to_process, desc="Summarize profile columns"):
            values = get_basic_dtype_ser(events[col]).values
            results = np.zeros(
                (grouping.ngroups, self.quantiles.shape[0]), dtype=np.float64
            )

            i = 0
            for _, idx in grp_idxs.items():
                cur_counts = counts[idx[0]]
                cur_vals = values[idx]
                if cur_counts < cur_vals.size:
                    raise ValueError(
                        "Number of observations exceeds the number of possible observations."
                    )
                results[i, :] = self._calc_sparse_quantiles_core(
                    n_obs=cur_counts,
                    nonzeros=cur_vals,
                    quantiles=self.quantiles,
                )
                i += 1

            all_results[col] = results

        # Build output DataFrame
        combined_results = np.concatenate(
            [all_results[col] for col in cols_to_process], axis=1
        )

        combined_columns = [
            f"{str(col)}_{str(q)}" for col in cols_to_process for q in self.quantiles
        ]

        grp_keys = grp_idxs.keys()
        if len(self.group_cols) == 1:
            quant_idx = pd.Index(grp_keys, name=self.group_cols[0])
        elif len(self.group_cols) > 1:
            quant_idx = pd.MultiIndex.from_tuples(grp_keys, names=self.group_cols)
        else:
            raise ValueError("At least one group column is required.")
        quantile_df = pd.DataFrame(
            data=combined_results,
            index=quant_idx,
            columns=combined_columns,
        )
        return quantile_df

    @staticmethod
    @jit
    def _calc_sparse_quantiles_core(
        n_obs: int,
        nonzeros: np.ndarray,
        quantiles: np.ndarray[np.float64],
    ) -> np.ndarray:
        """Calculate quantiles for sparse data by padding with zeros.

        Creates a full array with zeros and places the nonzero values at the end,
        then calculates quantiles. This approach correctly handles sparse data
        where many observations are zero.

        Args:
            n_obs (int): Total number of observations (including zeros).
            nonzeros (np.ndarray): Array of nonzero values.
            quantiles (np.ndarray[np.float64]): Array of quantile values to calculate.

        Returns:
            np.ndarray: Array of calculated quantile values.
        """
        arr = np.zeros(n_obs, dtype=nonzeros.dtype)
        n_nonzero = nonzeros.shape[0]
        arr[-n_nonzero:] = nonzeros
        qtls = np.quantile(a=arr, q=quantiles)
        return qtls
