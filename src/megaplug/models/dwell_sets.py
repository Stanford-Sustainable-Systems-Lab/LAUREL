"""Central data structure for vehicle dwell histories and dwell-level operations.

:class:`DwellSet` wraps a pandas or Dask DataFrame in which each row represents a
single dwell (parking stop) for a heavy-duty truck.  It enforces a canonical
sort order (vehicle, dwell-start time), manages a set of named semantic columns
(vehicle ID, hex cell, start/end times, trip distance/duration, reset flag), and
provides the core operations that downstream pipelines build on:

- **Masked accumulation** (:meth:`~DwellSet.accum_masked`): forward or reverse
  cumulative aggregation of trip-distance/duration across consecutive dwells
  that will be removed, respecting ``reset`` boundaries.  Used to propagate
  consumed-energy estimates from inserted optional stops back to their flanking
  depots.
- **Masked reset propagation** (:meth:`~DwellSet.reset_masked`): ensures that
  the first retained dwell after a gap introduced by masking has ``reset=True``,
  so that the charging-choice simulator restarts the SoC correctly.
- **Dwell → event conversion** (:meth:`~DwellSet.to_events`): reshapes from
  one-dwell-per-row to one-event-per-row for load profile construction.

Module-level helper functions :func:`load_dwell_set` and :func:`save_dwell_set`
provide the Kedro I/O interface.

Key design decisions
--------------------
- **Vehicle ID as DataFrame index**: The vehicle-ID column is stored as the
  DataFrame index rather than a regular column.  This lets Dask partition by
  vehicle so that per-vehicle groupby operations never need cross-partition
  communication.
- **Numba JIT inner loops**: The accumulation core
  (:meth:`~DwellSet._accum_masked_core`) and reset core
  (:meth:`~DwellSet._reset_masked_grp_core`) are decorated with ``@njit``.
  NumPy structured-array dtypes that are incompatible with Numba (booleans,
  nullable integers) are substituted via ``_replace_dtypes`` before the
  recarray is created.
- **``reset`` column semantics**: a ``True`` value at row ``i`` means "this
  dwell begins a new simulation epoch" — the charging simulator resets the
  vehicle's SoC to full at the start of each epoch.  The default behaviour
  (when no ``reset`` column is supplied) marks only the first dwell of each
  vehicle as a reset.
- **``CumAggFunc`` enum**: accumulation supports SUM, PRODUCT, MAX, and MIN so
  the same ``accum_masked`` method can propagate both additive quantities (trip
  distance, trip duration) and multiplicative ones.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.
"""

import copy
import logging
import re
from enum import IntEnum, auto
from itertools import product
from typing import Self

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from megaplug.utils.h3 import add_geometries

logger = logging.getLogger(__name__)


DEFAULT_COLUMN_NAMES = {
    "veh": "veh_id",
    "hex": "hex_id",
    "start": "dwell_start_time",
    "end": "dwell_end_time",
    "trip_dist": "trip_distance",
    "trip_dur": "trip_duration",
    "reset": "reset_state_before",
}


class CumAggFunc(IntEnum):
    """Aggregation function selector for :meth:`DwellSet.accum_masked`.

    Members:
        SUM: Accumulate by addition (``cur + cum_sum``).
        PRODUCT: Accumulate by multiplication (``cur * cum_sum``).
        MAX: Accumulate by element-wise maximum.
        MIN: Accumulate by element-wise minimum.
    """

    SUM = auto()
    PRODUCT = auto()
    MAX = auto()
    MIN = auto()


class DwellSet:
    """Container for vehicle dwell records supporting Dask-compatible operations.

    Each row of the underlying DataFrame represents one dwell (parking stop).
    The DataFrame index must be the vehicle-ID column; all other semantic
    columns (hex cell, start/end times, trip distance/duration, reset flag) are
    tracked by name via properties that transparently rename the underlying
    column when assigned.

    Class attributes:
        _replace_dtypes: Mapping from pandas/NumPy dtype names to Numba-safe
            equivalents used when constructing recarrays for JIT cores.
        is_dask: ``True`` when the underlying ``data`` is a Dask DataFrame;
            updated automatically by the ``data`` property setter.

    Typical usage::

        dw = DwellSet(
            data=df,
            veh="veh_id",
            hex="hex_id",
            start="dwell_start_time",
            end="dwell_end_time",
            trip_dist="trip_distance",
            trip_dur="trip_duration",
        )
    """

    _veh = None
    _veh_time = "veh_time_id"
    _hex = None
    _start = None
    _end = None
    _trip_dist = None
    _trip_dur = None
    _reset = None
    _seq_names = None
    _verify_sorting = None
    _replace_dtypes: dict = {np.bool.__name__: "u1"}
    data = None
    is_dask = False

    def __init__(
        self,
        data: dd.DataFrame | pd.DataFrame,
        veh: str,
        hex: str,
        start: str,
        end: str,
        trip_dist: str,
        trip_dur: str,
        reset: str | None = None,
        verify_sorting: bool = True,
        sorted: bool = False,
        *args,
        **kwargs,
    ):
        """Initialise a DwellSet from a pandas or Dask DataFrame.

        Validates that all named columns are present and unique, then sorts the
        data by ``(veh, start)`` (or skips sorting if ``sorted=True``).  If no
        ``reset`` column is provided, creates a default one that marks only the
        first dwell of each vehicle as a reset epoch.

        Args:
            data: DataFrame with one dwell per row.  The vehicle-ID column may
                be in the index or as a regular column; after construction it
                will always be the index.
            veh: Column name for vehicle identifiers.
            hex: Column name for H3 hex cell IDs at the dwell location.
            start: Column name for dwell start timestamps.
            end: Column name for dwell end timestamps.
            trip_dist: Column name for distance of the preceding trip (units
                consistent with the energy model, typically miles).
            trip_dur: Column name for duration of the preceding trip.
            reset: Column name for a boolean reset flag.  If ``None``, a default
                column named ``"reset_state_before"`` is created with ``True``
                only at the first dwell of each vehicle.
            verify_sorting: If ``True``, the :meth:`sort_by_veh_time` check is
                run before operations that require sorted order.  Defaults to
                ``True``.
            sorted: If ``True``, skip the initial sort and assume the data is
                already sorted by ``(veh, start)`` with ``veh`` as the index.
                Emits a warning because mis-ordered data will produce silently
                wrong results.  Defaults to ``False``.

        Raises:
            RuntimeError: If any two of the named columns share the same name,
                or if a named column is not found in the DataFrame.
        """
        self.data = data

        def has_duplicates(lst):
            return len(lst) != len(set(lst))

        dup_test = [veh, hex, start, end, trip_dist, trip_dur]
        dup = has_duplicates(dup_test)
        if dup:
            raise RuntimeError("Duplicated column names in arguments.")

        col_names = list(data.columns) + [data.index.name]

        def _return_if_present(name: str) -> str:
            if name in col_names:
                return name
            else:
                raise RuntimeError(f"{name} not found in data columns on indexes.")

        self._veh = _return_if_present(veh)
        self._hex = _return_if_present(hex)
        self._start = _return_if_present(start)
        self._end = _return_if_present(end)
        self._trip_dist = _return_if_present(trip_dist)
        self._trip_dur = _return_if_present(trip_dur)

        self.verify_sorting = verify_sorting

        if not sorted:
            if self.is_dask:
                # Includes index setting on vehicle id
                self.sort_by_veh_time(force=True)
            else:
                self.sort_by_veh_time()  # Allows for skipping sorting if already sorted
        else:
            logger.warning(
                "DwellSet is assumed to be sorted by (vehicle_id, time) with vehicle_id as index."
            )

        if reset is None:
            self._reset = DEFAULT_COLUMN_NAMES["reset"]
            self.set_default_reset_col()
        else:
            self._reset = _return_if_present(reset)

        self.sum_cols = [self.trip_dist, self.trip_dur]

    def copy_without_data(self: Self) -> Self:
        """Return a deep copy of the DwellSet with ``data`` set to ``None``.

        Used to clone column-name bindings and settings before attaching new
        underlying data (e.g. when filtering to a subset of vehicles).
        """
        new = copy.copy(self)
        new.data = None
        newer = copy.deepcopy(new)
        return newer

    def sort_by_veh_time(self, force: bool = False) -> None:
        """Sort ``data`` by ``(veh, start)`` and set ``veh`` as the index.

        Checks whether sorting is already correct unless ``force=True``.  For
        Dask DataFrames, always forces a sort because the check is expensive.

        Args:
            force: Skip the sorted-check and sort unconditionally.  Defaults to
                ``False``.
        """
        if not force:
            logger.info("Checking if DwellSet is sorted by vehicle and time.")
            is_sorted = self.is_sorted_by_veh_time()
            if is_sorted:
                logger.info("DwellSet is already sorted by vehicle and time.")
        if force or not is_sorted:
            logger.info("Sorting DwellSet by vehicle and time.")
            idx_name = self.data.index.name
            if idx_name in self.get_tracked_cols() and idx_name is not None:
                drop = False
            else:
                drop = True
            self.data = DwellSet._sort_by_grp_time(
                df=self.data, grp_col=self.veh, time_col=self.start, drop_cur_idx=drop
            )

    @staticmethod
    def _sort_by_grp_time(
        df: pd.DataFrame | dd.DataFrame,
        grp_col: str,
        time_col: str,
        drop_cur_idx: bool = False,
    ) -> pd.DataFrame | dd.DataFrame:
        """Sort ``df`` by ``(grp_col, time_col)`` and set ``grp_col`` as index.

        For Dask DataFrames, sets the index (repartitions by group) then sorts
        within each partition via ``groupby.apply``.  For pandas DataFrames,
        resets the current index (optionally dropping it), sorts on both
        columns, then sets the group column as the new index.

        Args:
            df: DataFrame to sort.
            grp_col: Column to use as grouping key and final index.
            time_col: Column to sort by within each group.
            drop_cur_idx: For pandas DataFrames, whether to discard the
                existing index during ``reset_index``.  Ignored for Dask.

        Returns:
            Sorted DataFrame with ``grp_col`` as the index.
        """
        if df.index.name == grp_col:
            grp_is_idx = True
            drop_cur_idx = False
        else:
            grp_is_idx = False

        if isinstance(df, dd.DataFrame):
            if not grp_is_idx:
                df = df.set_index(grp_col)
            else:
                pass  # Assuming that Dask index is sorted already
            df = df.groupby(grp_col, group_keys=False).apply(
                lambda grp: grp.sort_values(time_col), meta=dd.utils.make_meta(df)
            )
        elif isinstance(df, pd.DataFrame):
            df = df.reset_index(drop=drop_cur_idx)
            df = df.sort_values([grp_col, time_col])
            df = df.set_index(grp_col)

        return df

    def is_sorted_by_veh_time(self: Self) -> bool:
        """Return ``True`` if ``data`` is sorted by ``start`` within each vehicle."""
        return DwellSet._is_grp_time_structured(
            self.data,
            grp_col=self.veh,
            time_col=self.start,
        )

    @staticmethod
    def _is_grp_time_structured(
        df: pd.DataFrame | dd.DataFrame,
        grp_col: str,
        time_col: str,
    ) -> bool | np.ndarray:
        """Return ``True`` if ``df`` is monotonically sorted by ``time_col`` within each group.

        Requires ``grp_col`` to be the DataFrame index; returns ``False``
        immediately if it is not.  For Dask DataFrames, performs a
        ``map_partitions`` check assuming partitions respect group boundaries.

        Args:
            df: DataFrame to check.
            grp_col: Expected index column name.
            time_col: Time column to check for monotonic increase within groups.

        Returns:
            Boolean scalar (``False`` immediately) or boolean value derived
            from the per-group monotonicity check.
        """
        # If group column isn't index, then no good
        if df.index.name != grp_col:
            return False

        if isinstance(df, pd.DataFrame):
            # If that index isn't sorted, then no good
            grp_sorted = df.index.is_monotonic_increasing
            if not grp_sorted:
                return False
            # If time isn't sorted within groups, then no good
            time_sorted = DwellSet._groupby_increasing(
                df=df, grp_col=grp_col, sort_col=time_col
            )
        elif isinstance(df, dd.DataFrame):
            # Assuming that Dask index is group / sorted within partitions
            time_sorted = df.map_partitions(
                func=DwellSet._groupby_increasing,
                grp_col=grp_col,
                sort_col=time_col,
            )
        time_unsorted = ~time_sorted
        any_unsorted = time_unsorted.any()
        not_any_unsorted = ~any_unsorted
        return not_any_unsorted

    def _groupby_increasing(df: pd.DataFrame, grp_col: str, sort_col: str) -> pd.Series:
        """Return a per-group boolean Series indicating monotone increase of ``sort_col``."""
        time_sorted = df.groupby(grp_col)[sort_col].agg(
            lambda ser: ser.is_monotonic_increasing
        )
        return time_sorted

    def accum_masked(
        self,
        keep_mask_col: str,
        accum_cols: str | list[str] = None,
        reverse: bool | list[bool] = False,
        agg_func: CumAggFunc | list[CumAggFunc] = CumAggFunc.SUM,
        write_all: bool = False,
        inplace: bool = False,
    ) -> Self | None:
        """Accumulate trip quantities across consecutive dwells that will be removed.

        For each vehicle, iterates through dwells in chronological (or reverse)
        order and accumulates ``accum_cols`` values into the preceding (or
        following) *kept* dwell when the current dwell is masked out.  A
        ``reset`` boundary always restarts the accumulation counter.

        This is used, for example, to propagate the consumed-energy of an
        inserted optional truck-stop dwell back onto the depot dwell that
        precedes it, so that the charging simulator sees the correct total
        energy demand at each depot stop.

        The result is written to new columns named ``{col}_{keep_mask_col}``
        for each column in ``accum_cols``.  When ``write_all=False``, these
        new columns contain ``NaN``/``NA`` for rows where ``keep_mask_col`` is
        ``False`` (i.e. rows that would be dropped).

        .. note::
            Records for a vehicle that cross Dask partition boundaries will
            cause incorrect accumulation.  Ensure vehicles are fully contained
            within a single partition before calling this method.

        Args:
            keep_mask_col: Boolean column name; ``True`` means "keep this dwell".
            accum_cols: Column(s) to accumulate.  Defaults to
                ``[trip_dist, trip_dur, reset]`` if ``None``.
            reverse: If ``True``, accumulate backward (from end to start of each
                vehicle's dwell sequence).  May be a single bool or a per-column
                list of bools.  Defaults to ``False``.
            agg_func: Aggregation function(s) to apply.  May be a single
                :class:`CumAggFunc` or a per-column list.  Defaults to
                :attr:`CumAggFunc.SUM`.
            write_all: If ``True``, write accumulated values for both kept and
                dropped rows.  Defaults to ``False``.
            inplace: If ``True``, modify ``self.data`` in place and return
                ``None``; otherwise return a deep copy.  Defaults to ``False``.

        Returns:
            Modified :class:`DwellSet` (or ``None`` if ``inplace=True``), with
            new ``{col}_{keep_mask_col}`` columns added to ``data``.

        Raises:
            ValueError: If ``reverse`` or ``agg_func`` are lists whose length
                does not match ``accum_cols``.
        """
        if self.verify_sorting:
            self.sort_by_veh_time()

        if inplace:
            new = self
        else:
            new = copy.deepcopy(self)

        if accum_cols is None:
            accum_cols = [self.trip_dist, self.trip_dur, self.reset]

        kws = dict(
            keep_mask_col=keep_mask_col,
            accum_cols=accum_cols,
            reset_col=self.reset,
            veh_col=self.veh,
            replace_dtypes=self._replace_dtypes,
            reverse=reverse,
            agg_func=agg_func,
            write_all=write_all,
        )
        if self.is_dask:
            new.data = new.data.map_partitions(
                self._accum_masked_df, show_progress=False, **kws
            )
        else:
            new.data = self._accum_masked_df(df=new.data, **kws)

        if inplace:
            return None
        else:
            return new

    @staticmethod
    def _accum_masked_df(
        df: pd.DataFrame,
        keep_mask_col: str,
        accum_cols: str | list[str],
        reset_col: str,
        veh_col: str,
        replace_dtypes: dict[str, str],
        reverse: bool | list[bool],
        agg_func: CumAggFunc | list[CumAggFunc],
        write_all: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Apply :meth:`_accum_masked_core` per vehicle on a pandas DataFrame.

        Converts the keep-mask and reset columns to a NumPy recarray (using
        ``replace_dtypes`` to substitute Numba-incompatible types), then
        iterates over per-vehicle index groups and calls the JIT core.  Writes
        accumulated values into new columns and optionally nulls out rows that
        will be dropped.

        Args:
            df: Input pandas DataFrame sorted by ``(veh_col, time)``.
            keep_mask_col: Boolean column name.
            accum_cols: Column(s) to accumulate.
            reset_col: Boolean column indicating epoch boundaries.
            veh_col: Vehicle-ID column (used for ``groupby`` index lookup).
            replace_dtypes: Dtype substitution map for recarray construction.
            reverse: Backward-accumulation flag(s).
            agg_func: Aggregation function(s).
            write_all: Whether to also write values for dropped rows.
            show_progress: Whether to show a ``tqdm`` progress bar.

        Returns:
            ``df`` with new ``{col}_{keep_mask_col}`` columns added.
        """
        if isinstance(accum_cols, str):
            accum_cols = [accum_cols]
        if isinstance(reverse, bool):
            reverse = [reverse] * len(accum_cols)
        elif isinstance(reverse, list) and (len(reverse) != len(accum_cols)):
            raise ValueError(
                "'reverse' must be a single bool or a list of bools of the same length as 'accum_cols'."
            )
        if isinstance(agg_func, CumAggFunc):
            agg_func = [agg_func] * len(accum_cols)
        elif isinstance(agg_func, list) and (len(agg_func) != len(accum_cols)):
            raise ValueError(
                "'agg_func' must be a single CumAggFunc or a list of CumAggFuncs of the same length as 'accum_cols'."
            )

        logic_renamer = {keep_mask_col: "keep", reset_col: "reset"}
        logic_df = df.loc[:, list(logic_renamer.keys())]
        logic_df = logic_df.rename(columns=logic_renamer)
        logic_dtypes = DwellSet._get_recarray_dtypes(logic_df, replace_dtypes)
        logic_recs = logic_df.to_records(column_dtypes=logic_dtypes, index=False)

        vals = {col: df[col].values for col in accum_cols}
        outs = {col: np.zeros_like(vals[col]) for col in accum_cols}

        # Using pandas groupby indices to move over dwell recarray
        grp_idxs = df.groupby(veh_col).indices
        col_rev = list(zip(accum_cols, reverse, agg_func))

        itr = grp_idxs.items()
        if show_progress:
            itr = tqdm(itr)

        for grp, idxs in itr:
            logics = logic_recs[idxs]
            for col, rev, fnc in col_rev:
                outs[col][idxs] = DwellSet._accum_masked_core(
                    logics=logics,
                    vals=vals[col][idxs],
                    outs=outs[col][idxs],
                    reverse=rev,
                    agg_func=fnc,
                )

        # Build output dataframe
        def _get_new_col_name(col: str) -> str:
            return f"{col}_{keep_mask_col}"

        for col in accum_cols:
            new_col = _get_new_col_name(col)
            df.loc[:, new_col] = outs[col]
            is_float_col = np.issubdtype(df[new_col].dtype, np.floating)
            if not is_float_col:
                df[new_col] = df[new_col].convert_dtypes()
                null_val = pd.NA
            else:
                null_val = np.nan

            if not write_all:
                df.loc[~df[keep_mask_col], new_col] = null_val

        return df

    @staticmethod
    def _get_recarray_dtypes(
        df: pd.DataFrame, replace_dtypes: dict[str, str]
    ) -> dict[str, str]:
        """Build a column-dtype dict suitable for ``DataFrame.to_records``.

        Substitutes any dtype whose name appears in ``replace_dtypes`` (e.g.
        NumPy ``bool`` → ``"u1"``) so the resulting recarray is compatible with
        Numba JIT functions.

        Args:
            df: DataFrame whose column dtypes will be inspected.
            replace_dtypes: Mapping from incompatible dtype names to replacement
                dtype strings.

        Returns:
            Dict mapping column names to dtype strings for columns that require
            substitution; other columns are not included (``to_records`` will
            use their native dtypes).
        """
        col_dtypes = {}
        for col, dtype in zip(df.columns, df.dtypes):
            if dtype.name == np.bool.__name__:
                col_dtypes.update({col: replace_dtypes[dtype.name]})
        return col_dtypes

    @staticmethod
    @njit
    def _accum_masked_core(
        logics: np.recarray,
        vals: np.ndarray,
        outs: np.ndarray,
        reverse: bool = False,
        agg_func: CumAggFunc = CumAggFunc.SUM,
    ) -> np.ndarray:
        """JIT-compiled inner loop for masked accumulation over a single vehicle.

        Iterates through ``vals`` (forward or backward) and applies
        ``agg_func`` to accumulate values from dropped rows onto the next
        retained row.  When a reset boundary is encountered, the running
        accumulator is reset to the current raw value instead.

        Args:
            logics: Structured NumPy array with fields ``"keep"`` (uint8 bool)
                and ``"reset"`` (uint8 bool), one element per dwell.
            vals: 1-D array of the quantity to accumulate, one element per
                dwell.
            outs: Pre-allocated output array of the same shape as ``vals``.
            reverse: If ``True``, iterate from last to first dwell.
            agg_func: :class:`CumAggFunc` member controlling how values are
                combined.  Defaults to :attr:`CumAggFunc.SUM`.

        Returns:
            ``outs`` filled with accumulated values.
        """
        nsteps = logics.shape[0]
        if not nsteps == vals.shape[0] == outs.shape[0]:
            raise RuntimeError("The three arrays must have the same length.")

        cum_sum = 0
        prev_reset = False
        if reverse:
            itr = range(-1, -(nsteps + 1), -1)
        else:
            itr = range(nsteps)
        for i in itr:
            if (not reverse and logics["reset"][i]) or (reverse and prev_reset):
                cur = vals[i]  # With reset, we just copy the original
            else:  # With no reset, we apply accumulation
                match agg_func:
                    case CumAggFunc.SUM:
                        cur = vals[i] + cum_sum
                    case CumAggFunc.PRODUCT:
                        cur = vals[i] * cum_sum
                    case CumAggFunc.MAX:
                        cur = np.maximum(vals[i], cum_sum)
                    case CumAggFunc.MIN:
                        cur = np.minimum(vals[i], cum_sum)

            prev_reset = logics["reset"][i]

            if logics["keep"][i]:  # If we're keeping this row, then reset the cumsum
                cum_sum = 0
            else:
                cum_sum = cur

            outs[i] = cur
        return outs

    def drop_masked(
        self, keep_mask_col: str, inplace: bool = False, drop_mask_col: bool = True
    ) -> Self | None:
        """Remove rows where ``keep_mask_col`` is ``False`` or ``NA``.

        Converts ``keep_mask_col`` to nullable boolean, replaces ``False``
        with ``pd.NA``, and calls ``dropna``.  Optionally removes the mask
        column from the result.

        Args:
            keep_mask_col: Boolean column; rows with ``True`` are retained.
            inplace: Modify in place if ``True``, return deep copy otherwise.
            drop_mask_col: Drop ``keep_mask_col`` from the result.  Defaults
                to ``True``.

        Returns:
            Modified :class:`DwellSet` (or ``None`` if ``inplace=True``).
        """
        if inplace:
            new = self
        else:
            new = copy.deepcopy(self)

        kws = {}
        if not self.is_dask:
            kws.update(dict(inplace=True))

        new.data[keep_mask_col] = (
            new.data[keep_mask_col].astype("boolean").replace(False, pd.NA)
        )
        new.data.dropna(subset=keep_mask_col, **kws)
        if drop_mask_col:
            new.data.drop(columns=[keep_mask_col], **kws)

        if inplace:
            return None
        else:
            return new

    def reset_masked(self, keep_mask_col: str, inplace: bool = False) -> Self | None:
        """Set ``reset=True`` on the first kept dwell after each removed gap.

        After masking out dwells, the first retained dwell that immediately
        follows one or more removed dwells must be flagged as a reset so the
        charging simulator restarts the SoC correctly.  This method writes a
        new column ``"{reset}_{keep_mask_col}"`` with the updated reset flags.

        The JIT core (:meth:`_reset_masked_grp_core`) is pre-compiled with a
        small dummy array before the main loop to avoid Numba cold-start
        overhead.

        Args:
            keep_mask_col: Boolean column; ``True`` means "keep this dwell".
            inplace: Modify in place if ``True``, return deep copy otherwise.
                Defaults to ``False``.

        Returns:
            Modified :class:`DwellSet` (or ``None`` if ``inplace=True``), with
            new column ``"{reset}_{keep_mask_col}"`` added to ``data``.
        """
        if self.verify_sorting:
            self.sort_by_veh_time()

        # Force numba compilation
        base = np.array(
            [True, False, True]
        )  # Just an example array of the correct dtype
        _ = DwellSet._reset_masked_grp_core(
            keep=base,
            reset=base,
        )

        if inplace:
            new = self
        else:
            new = copy.deepcopy(self)

        # Pre-allocate target column
        new.data.loc[:, f"{self.reset}_{keep_mask_col}"] = False

        kws = dict(
            func=DwellSet._reset_masked_grp,
            keep_mask_col=keep_mask_col,
            reset_col=self.reset,
        )
        if self.is_dask:
            kws.update(dict(meta=dd.utils.make_meta(self.data)))
            new.data = self.data.groupby(self.veh, group_keys=False, sort=False).apply(
                **kws
            )
        else:
            tqdm.pandas()
            new.data = self.data.groupby(
                self.veh, group_keys=False, sort=False
            ).progress_apply(**kws)
        if inplace:
            return None
        else:
            return new

    @staticmethod
    def _reset_masked_grp(
        grp: pd.DataFrame, keep_mask_col: str, reset_col: str
    ) -> pd.DataFrame:
        """Apply :meth:`_reset_masked_grp_core` to a single vehicle group.

        Writes updated reset flags into ``"{reset_col}_{keep_mask_col}"``.

        Args:
            grp: Single-vehicle pandas DataFrame partition.
            keep_mask_col: Boolean column indicating retained rows.
            reset_col: Existing reset column whose values will be updated.

        Returns:
            ``grp`` with the new reset column added.
        """
        new_name = f"{reset_col}_{keep_mask_col}"
        grp.loc[:, new_name] = DwellSet._reset_masked_grp_core(
            keep=grp[keep_mask_col].values,
            reset=grp[reset_col].values,
        )
        return grp

    @staticmethod
    @njit
    def _reset_masked_grp_core(keep: np.ndarray, reset: np.ndarray) -> np.ndarray:
        """JIT-compiled core: set ``reset=True`` at each transition from dropped to kept.

        Scans ``keep`` and sets the corresponding ``reset`` element to ``True``
        whenever the current row is kept and the previous row was not (i.e. the
        start of a new retained segment), provided the row was not already a
        reset boundary.

        Args:
            keep: 1-D boolean array (uint8); ``1`` = keep this dwell.
            reset: 1-D boolean array (uint8) of existing reset flags.  Modified
                in place.

        Returns:
            Updated ``reset`` array.

        Raises:
            RuntimeError: If ``keep`` and ``reset`` have different shapes.
        """
        if not keep.shape == reset.shape:
            raise RuntimeError("The two arrays must have the same shape.")
        prev_keep = False
        for i in range(keep.shape[0]):
            start_keep = np.logical_and(keep[i], np.logical_not(prev_keep))
            prev_keep = keep[i]
            if start_keep and not reset[i]:
                reset[i] = True
        return reset

    def set_default_reset_col(self):
        """Initialise ``data[reset]`` so only each vehicle's first dwell is ``True``.

        Creates the ``reset`` column with all-``False`` values, then uses a
        per-vehicle groupby to flip the first row of each group to ``True``.
        """
        self.data[self.reset] = False
        if self.is_dask:
            self.data = self.data.groupby(self.veh, group_keys=False, sort=False).apply(
                DwellSet._set_default_reset_col_grp,
                reset_col=self.reset,
                include_groups=False,
                meta=dd.utils.make_meta(self.data),
            )
        else:
            tqdm.pandas()
            self.data = self.data.groupby(
                self.veh, group_keys=False, sort=False
            ).progress_apply(
                DwellSet._set_default_reset_col_grp,
                reset_col=self.reset,
                include_groups=False,
            )

    @staticmethod
    def _set_default_reset_col_grp(grp: pd.DataFrame, reset_col: str) -> pd.DataFrame:
        reset_col_idx = grp.columns.get_loc(reset_col)
        grp.iat[0, reset_col_idx] = True
        return grp

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if value is not None:
            if isinstance(value, dd.DataFrame):
                self.is_dask = True
            elif isinstance(value, pd.DataFrame | gpd.GeoDataFrame):
                self.is_dask = False
            else:
                raise NotImplementedError(
                    "DwellSet's data must be a Dask or Pandas DataFrame."
                )
        self._data = value

    @property
    def veh(self):
        return self._veh

    @veh.setter
    def veh(self, value):
        self._rename_idx_col(value, self._veh)
        self._veh = value

    @property
    def hex(self):
        return self._hex

    @hex.setter
    def hex(self, value):
        self._rename_idx_col(value, self._hex)
        self._hex = value

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        self._rename_idx_col(value, self._start)
        self._start = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        self._rename_idx_col(value, self._end)
        self._end = value

    @property
    def trip_dist(self):
        return self._trip_dist

    @trip_dist.setter
    def trip_dist(self, value):
        self._rename_idx_col(value, self._trip_dist)
        self._trip_dist = value

    @property
    def trip_dur(self):
        return self._trip_dur

    @trip_dur.setter
    def trip_dur(self, value):
        self._rename_idx_col(value, self._trip_dur)
        self._trip_dur = value

    @property
    def reset(self):
        return self._reset

    @reset.setter
    def reset(self, value):
        self._rename_idx_col(value, self._reset)
        self._reset = value

    @property
    def verify_sorting(self):
        return self._verify_sorting

    @verify_sorting.setter
    def verify_sorting(self, value):
        self._verify_sorting = value

    @property
    def seq_names(self):
        return self._seq_names

    @seq_names.setter
    def seq_names(self, value):
        # Check that all values in seq_names are used at least once as a prefix, else issue a warning
        if not (isinstance(value, list | tuple)):
            raise RuntimeError(
                "Sequence names must be passed as a non-string iterable, like a list or tuple."
            )
        col_names = self.data.columns + [self.data.index.name]
        for s in value:
            present = any(s in col for col in col_names)
            if not present:
                raise RuntimeError(f"Sequence name {s} not found in columns.")
        self._seq_names = value

    def _rename_idx_col(self, new: str, old: str):
        if self.data.index.name == old:
            self.data.index = self.data.index.rename(new)
        else:
            self.data = self.data.rename(columns={old: new})

    def get_tracked_cols(self):
        """Return a list of all semantic column/index names tracked by this DwellSet."""
        return [
            self._veh,
            self._hex,
            self._start,
            self._end,
            self._trip_dist,
            self._trip_dur,
            self._reset,
        ]

    @classmethod
    def from_trips(
        cls,
        trips: dd.DataFrame,
        veh: str,
        hex: str,
        start_trip: str,
        end_trip: str,
        trip_dist: str,
        trip_dur: str,
        verify_sorting: bool = True,
        sorted: bool = False,
        *args,
        **kwargs,
    ):
        """Construct a DwellSet from trip-formatted data.

        Interprets each row as a trip arrival, so the dwell starts at
        ``end_trip`` and ends at the ``start_trip`` of the *next* trip for the
        same vehicle (computed by a within-group shift).  The last trip of each
        vehicle is dropped because no subsequent start time exists.

        After construction the dwell start/end columns are renamed to the
        canonical ``DEFAULT_COLUMN_NAMES["start"]`` and ``["end"]`` values.

        Args:
            trips: DataFrame with one trip per row.
            veh: Vehicle-ID column name.
            hex: Dwell-location hex column name.
            start_trip: Trip departure timestamp column.
            end_trip: Trip arrival timestamp column (becomes dwell start).
            trip_dist: Preceding-trip distance column.
            trip_dur: Preceding-trip duration column.
            verify_sorting: Passed through to :meth:`__init__`.
            sorted: Passed through to :meth:`__init__`.

        Returns:
            New :class:`DwellSet` with canonical start/end column names.
        """
        dw = cls(
            data=trips,
            veh=veh,
            hex=hex,
            start=end_trip,
            end=start_trip,  # This is almost true, since the dwell end is the shifted start_trip
            trip_dist=trip_dist,
            trip_dur=trip_dur,
            verify_sorting=verify_sorting,
        )

        def _shift_by_grp(grp: pd.DataFrame, src: str, tgt: str) -> pd.DataFrame:
            grp[tgt] = grp[src].shift(-1)
            return grp

        if dw.is_dask:
            dw.data = dw.data.groupby(dw.veh, group_keys=False).apply(
                _shift_by_grp,
                src=dw.end,
                tgt=dw.end,
                meta=dd.utils.make_meta(dw.data),
            )
        else:
            dw.data = dw.data.groupby(dw.veh, group_keys=False).apply(
                _shift_by_grp,
                src=dw.end,
                tgt=dw.end,
            )

        dw.data = dw.data.dropna(subset=dw.end)  # To get rid of NaT value from shift

        non_idx_cols = [
            col for col in dw.get_tracked_cols() if col != dw.data.index.name
        ]
        extra_cols = [col for col in dw.data.columns if col not in non_idx_cols]
        dw.data = dw.data.loc[:, non_idx_cols + extra_cols]
        dw.start = DEFAULT_COLUMN_NAMES["start"]
        dw.end = DEFAULT_COLUMN_NAMES["end"]
        return dw

    def to_events(self: Self, id_cols: list[str] = None) -> dd.DataFrame | pd.DataFrame:
        """Reshape from one-dwell-per-row to one-event-per-row format.

        Uses :attr:`seq_names` to identify column groups (e.g.
        ``"dwell_start_time"``, ``"dwell_start_power_kw"`` share the prefix
        ``"dwell_start"``).  Each named sequence becomes a separate row in the
        output, stacked via a MultiIndex column pivot then ``stack``.

        :attr:`seq_names` must be set before calling this method.

        Args:
            id_cols: Columns to retain as identifiers in the output (default:
                ``[veh, hex]``).

        Returns:
            DataFrame with one event per row, indexed by ``event_id``, and
            columns matching the suffix components of the sequence columns.

        Raises:
            RuntimeError: If :attr:`seq_names` is ``None``.
        """
        if self.seq_names is None:
            raise RuntimeError("Sequence names must be set before calling 'to_events'.")

        drop_idx = False if self.data.index.name in self.get_tracked_cols() else True
        kws = dict(
            id_cols=id_cols if id_cols is not None else [self.veh, self.hex],
            seq_names=self.seq_names,
            drop_cur_idx=drop_idx,
        )
        if self.is_dask:
            events = self.data.map_partitions(DwellSet._dwells_to_events_grp, **kws)
        else:
            events = DwellSet._dwells_to_events_grp(dw=self.data, **kws)
        return events

    @staticmethod
    def _dwells_to_events_grp(
        dw: pd.DataFrame,
        id_cols: list[str],
        seq_names: list[str],
        drop_cur_idx: bool = False,
    ) -> pd.DataFrame:
        """Pivot one partition from dwell-wide to event-long format.

        For each sequence name in ``seq_names``, finds all columns whose name
        contains that prefix, extracts the tail (the part after
        ``"{seq_name}_"``), builds a MultiIndex column structure, and stacks
        on ``seq_id``/``seq_name`` to produce one row per (dwell × sequence
        event).

        Args:
            dw: Single pandas DataFrame partition, one row per dwell.
            id_cols: Identifier columns preserved in the output.
            seq_names: Column-prefix strings identifying each event sequence
                within a dwell (e.g. ``["dwell_start", "dwell_end"]``).
            drop_cur_idx: Whether to drop the current index during
                ``reset_index``.

        Returns:
            Long-format DataFrame indexed by ``event_id``, with one row per
            sequence event per dwell.
        """
        # Set new column MultiIndex to prepare for stacking
        dw = dw.reset_index(drop=drop_cur_idx)
        dw.set_index(id_cols, inplace=True, append=True)
        keep_cols = []
        tups = []
        for i, s in enumerate(seq_names):
            orig = [col for col in dw.columns if s in col]
            tails = [DwellSet._get_seq_name_tail(s, c) for c in orig]
            tups.extend(product(tails, [s], [i]))
            keep_cols.extend(orig)
        idx = pd.MultiIndex.from_tuples(tups, names=["variable", "seq_name", "seq_id"])
        dw = dw[keep_cols]
        dw.columns = idx

        # Stack the dwells into events
        events = dw.stack(level=["seq_id", "seq_name"], future_stack=True)
        events = events.reset_index()
        events.index = events.index.rename("event_id")
        drop_cols = list(set(events.columns) - set(id_cols + tails))
        events = events.drop(columns=drop_cols)
        events.columns.name = None
        return events

    @staticmethod
    def _get_seq_name_tail(seq_name: str, col_name: str) -> str:
        """Get the value part of a sequence name."""
        matches = re.findall(f"(?<={seq_name}_).+", col_name)
        if len(matches) == 0:
            raise RuntimeError("The column does not include the desired sequence name.")
        return matches[0]

    def to_geodataframe(self, geom_type: str = "point") -> None:
        """Attach H3-derived geometries to ``data``, converting it to a GeoDataFrame.

        Delegates to :func:`~megaplug.utils.h3.add_geometries` using the hex
        column.  Mutates ``data`` in place.

        Args:
            geom_type: Geometry type passed to ``add_geometries`` —
                ``"point"`` for hex centroids or ``"polygon"`` for hex
                boundaries.  Defaults to ``"point"``.
        """
        self.data = add_geometries(self.data, hex_col=self.hex, geom_type=geom_type)


def load_dwell_set(dwells: pd.DataFrame, params: dict) -> DwellSet:
    """Construct a DwellSet from a Kedro-loaded DataFrame and parameter dict.

    Passes all keys in ``params`` as keyword arguments to :class:`DwellSet`,
    so ``params`` must at minimum supply ``veh``, ``hex``, ``start``, ``end``,
    ``trip_dist``, and ``trip_dur``.

    Args:
        dwells: Raw DataFrame loaded by Kedro.
        params: Dict of column-name and option arguments for :class:`DwellSet`.

    Returns:
        Configured :class:`DwellSet` instance.
    """
    dw = DwellSet(data=dwells, **params)
    return dw


def save_dwell_set(dw: DwellSet) -> pd.DataFrame:
    """Extract the underlying DataFrame from a DwellSet for Kedro persistence.

    Args:
        dw: Populated :class:`DwellSet`.

    Returns:
        The ``data`` attribute ready for Kedro to save.
    """
    return dw.data
