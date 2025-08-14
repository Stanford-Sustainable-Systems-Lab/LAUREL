import datetime
from typing import Self

import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm

from megaPLuG.utils.data import IndexIntegerizer, get_basic_dtype_ser


class EventExpander:
    """Expand a dataframe of events with durations so that the value column is copied
    to one timestamp in each time block (defined by frequency) which it intersects.

    Attributes:
        time_col (str): Name of the column containing timestamps.
        dur_col (str): Name of the column containing event durations.
        value_col (str): Name of the column containing values to be expanded.
        group_cols (list[str]): List of column names to group by.
        freq (str): Frequency string for time block definition (e.g., '1H', '1D').
    """

    time_col: str
    dur_col: str
    value_col: str
    group_cols: list[str]
    freq: str

    def __init__(
        self: Self,
        time_col: str,
        dur_col: str,
        value_col: str,
        group_cols: list[str],
        freq: str,
    ) -> None:
        """Initialize the EventExpander.

        Args:
            time_col (str): Name of the column containing timestamps.
            dur_col (str): Name of the column containing event durations.
            value_col (str): Name of the column containing values to be expanded.
            group_cols (list[str]): List of column names to group by.
            freq (str): Frequency string for time block definition (e.g., '1H', '1D').
        """
        self.time_col = time_col
        self.dur_col = dur_col
        self.value_col = value_col
        self.group_cols = group_cols
        self.freq = freq

    def expand_events(
        self: Self, events: pd.DataFrame, return_expansions_only: bool = False
    ) -> pd.DataFrame:
        """Expand out events to cover all intermediate time units given their start and
        end timestamps.

        Args:
            events (pd.DataFrame): DataFrame containing events with time, duration, and
                value columns.
            return_expansions_only (bool, optional): If True, return only the rows \
                corresponding to expanded events. If False, return rows for both
                expanded and non-expanded events. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with expanded events, containing group columns,
                time column, and value column.

        Raises:
            RuntimeError: If the source time column is not timezone-naive or UTC.
        """
        source_tz = events[self.time_col].dt.tz
        if source_tz == datetime.UTC:
            send_to_utc = False  # If source time is UTC, then it is already compatible
        elif source_tz is None:
            send_to_utc = True
        else:
            raise RuntimeError("The source time column must be time zone naïve or UTC.")

        end_of_time_group = events[self.time_col].dt.ceil(self.freq)
        end_of_event = events[self.time_col] + events[self.dur_col]
        is_overflow = end_of_time_group < end_of_event
        need_expansion = events.loc[is_overflow]

        if need_expansion.index.names != [None]:
            need_expansion = need_expansion.reset_index()
        need_expansion = need_expansion.set_index(self.group_cols)
        group_inter = IndexIntegerizer(int_col="codes")
        need_expansion = group_inter.integerize(need_expansion)
        need_expansion = need_expansion.reset_index()
        if send_to_utc:
            need_expansion[self.time_col] = need_expansion[
                self.time_col
            ].dt.tz_localize(datetime.UTC)

        expanded = self._expand_events_wrapper(need_expansion)

        if send_to_utc:
            expanded[self.time_col] = expanded[self.time_col].dt.tz_localize(None)
        expanded = expanded.set_index("codes")
        expanded = group_inter.deintegerize(expanded)
        expanded = expanded.reset_index()

        if return_expansions_only:
            return expanded
        else:
            keep_cols = self.group_cols + [self.time_col, self.value_col]
            not_expanded = events.loc[:, keep_cols]
            all_nonzero = pd.concat([expanded, not_expanded], axis=0, ignore_index=True)
            return all_nonzero

    def _expand_events_wrapper(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert pandas Series to numpy arrays and set dtypes for self._expand_events_core().

        Args:
            df (pd.DataFrame): DataFrame with events that need expansion, must contain
                time_col, dur_col, value_col, and 'codes' columns.

        Returns:
            pd.DataFrame: DataFrame with expanded events containing 'codes', time_col,
                and value_col columns.
        """
        orig_time_type = df[self.time_col].dtype
        starts = df[self.time_col].dt.floor(self.freq)
        ends = (df[self.time_col] + df[self.dur_col]).dt.floor(self.freq)

        out_grps, out_times, out_vals = self._expand_events_core(
            starts=starts.values.astype(np.int64),
            ends=ends.values.astype(np.int64),
            vals=get_basic_dtype_ser(df[self.value_col]).values,
            grps=get_basic_dtype_ser(df["codes"]).values,
            tstep_ns=pd.Timedelta(self.freq).value,
        )

        out_times = pd.to_datetime(out_times.astype("datetime64[ns]"), utc=True)
        out_times = out_times.astype(orig_time_type)
        out = pd.DataFrame(
            {
                "codes": out_grps,
                self.time_col: out_times,
                self.value_col: out_vals,
            }
        )
        out = out.convert_dtypes()
        return out

    @staticmethod
    @jit
    def _expand_events_core(
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
            vals (np.ndarray): Array of values to be expanded.
            grps (np.ndarray): Array of group identifiers.
            tstep_ns (int): Time step in nanoseconds.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
                - grps_exp: Expanded group identifiers
                - times_exp: Expanded timestamps
                - vals_exp: Expanded values
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
    """Summarize dataframe groups which only contain the nonzero elements. Uses a
    correspondence table to determine how many zeros to include in the summary
    calculations.

    Attributes:
        group_cols (list[str]): List of column names to group by.
        quantiles (np.ndarray): Array of quantile values to calculate (e.g., [0.25, 0.5, 0.75]).
    """

    group_cols: list[str]
    quantiles: np.ndarray

    def __init__(self: Self, group_cols: list[str], quantiles: np.ndarray) -> None:
        """Initialize the NonzeroGroupedSummarizer.

        Args:
            group_cols (list[str]): List of column names to group by.
            quantiles (np.ndarray): Array of quantile values to calculate (e.g., [0.25, 0.5, 0.75]).
        """
        self.group_cols = group_cols
        self.quantiles = quantiles

    def summarize(
        self: Self,
        events: pd.DataFrame,
        value_col: str,
        possible_count_col: str,
    ) -> pd.DataFrame:
        """Calculate quantiles using observations paired with the count of possible
        observations to represent zeros.

        Args:
            events (pd.DataFrame): DataFrame containing the events to summarize.
            value_col (str): Name of the column containing the values to calculate quantiles for.
            possible_count_col (str): Name of the column containing the total count of
                possible observations (including zeros).

        Returns:
            pd.DataFrame: DataFrame with quantiles calculated for each group, indexed by
                group columns and with quantile values as column names.

        Raises:
            ValueError: If the number of observations exceeds the number of possible observations.
        """
        grouping = events.groupby(self.group_cols, observed=True)
        grp_idxs = grouping.indices
        values = get_basic_dtype_ser(events[value_col]).values
        counts = get_basic_dtype_ser(events[possible_count_col]).values
        results = np.zeros(
            (grouping.ngroups, self.quantiles.shape[0]), dtype=np.float64
        )

        i = 0
        for _, idx in tqdm(grp_idxs.items()):
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
        quantile_df = pd.DataFrame(
            data=results, index=grp_idxs.keys(), columns=self.quantiles
        )
        quantile_df.index.names = self.group_cols
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
