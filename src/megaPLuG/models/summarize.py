import datetime
from typing import Self

import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm

from megaPLuG.utils.data import IndexIntegerizer, get_basic_dtype_ser


class IntervalBeginSpreader:
    """Spread time-interval observations onto the interval beginnings of a given
    frequency which they cover. This is achieved by creating new observations with the
    same value as the original observation at these covered interval-beginnings.

    Attributes:
        time_col (str): Name of the column containing timestamps.
        dur_col (str): Name of the column containing observation durations.
        value_cols (list[str]): List of column names containing values to be spread.
        group_cols (list[str]): List of column names to group by.
        freq (str): Frequency string for time block definition (e.g., '1H', '1D').
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
        """Spread observations to cover all intermediate time units given their start
        times and durations.

        Args:
            obs (pd.DataFrame): DataFrame containing observations with time, duration, and
                value columns.
            return_spreaded_only (bool, optional): If True, return only the rows \
                corresponding to the "spreaded" observations. If False, return rows for both
                original and spreaded observations. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with expanded observations, containing group columns,
                time column, and all value columns with original data types preserved.

        Raises:
            RuntimeError: If the source time column is not timezone-naive or UTC.
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
        """Convert pandas Series to numpy arrays and set dtypes for self._spread_core().

        Args:
            df (pd.DataFrame): DataFrame with events that need expansion, must contain
                time_col, dur_col, value_cols, and 'codes' columns.

        Returns:
            pd.DataFrame: DataFrame with expanded events containing 'codes', time_col,
                and all value_cols with original data types preserved.
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
    """Summarize dataframe groups which only contain the nonzero elements. Uses a
    correspondence table to determine how many zeros to include in the summary
    calculations.

    Attributes:
        group_cols (list[str]): List of column names to group by.
        quantiles (np.ndarray): Array of quantile values to calculate (e.g., [0.25, 0.5, 0.75]).
        value_cols (list[str] | None): List of column names containing values to calculate quantiles for.
            If None, value columns must be specified in summarize() method.
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
        """Calculate quantiles using observations paired with the count of possible
        observations to represent zeros.

        Args:
            events (pd.DataFrame): DataFrame containing the events to summarize.
            value_cols (list[str] | str): Name(s) of column(s) containing values to calculate
                quantiles for. Can be a single string or list of strings.
            possible_count_col (str): Name of the column containing the total count of
                possible observations (including zeros).

        Returns:
            pd.DataFrame: DataFrame with quantiles calculated for each group. For single column,
                indexed by group columns with quantile values as column names. For multiple columns,
                indexed by group columns with MultiIndex columns (value_col_name, quantile_value).

        Raises:
            ValueError: If the number of observations exceeds the number of possible observations.
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
        if len(cols_to_process) == 1:
            # Single column: maintain backward compatibility
            col = cols_to_process[0]
            quantile_df = pd.DataFrame(
                data=all_results[col], index=grp_idxs.keys(), columns=self.quantiles
            )
            quantile_df.index.names = self.group_cols
        else:
            # Multiple columns: use MultiIndex columns
            multi_columns = pd.MultiIndex.from_product(
                [cols_to_process, self.quantiles], names=["value_col", "quantile"]
            )

            # Concatenate results horizontally
            combined_results = np.concatenate(
                [all_results[col] for col in cols_to_process], axis=1
            )

            quantile_df = pd.DataFrame(
                data=combined_results, index=grp_idxs.keys(), columns=multi_columns
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
