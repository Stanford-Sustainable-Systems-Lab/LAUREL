import copy
import logging
import re
from itertools import product
from typing import Self

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from megaPLuG.utils.h3 import add_geometries
from numba import njit
from numba.extending import overload
from tqdm import tqdm

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


class DwellSet:
    """The DwellSet represents tours taken by one or more vehicles.

    We use as many `dask`-compatible functions as possible to ease scalability. As such,
    we pay special attention to the indices and ordering.
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
        *args,
        **kwargs,
    ):
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
        if reset is None:
            self._reset = DEFAULT_COLUMN_NAMES["reset"]
            self.set_default_reset_col()
        else:
            self._reset = _return_if_present(reset)

        self.verify_sorting = verify_sorting
        self.sum_cols = [self.trip_dist, self.trip_dur]

    def copy_without_data(self: Self) -> Self:
        """Copy the DwellSet without its underlying data. This is used for filtering."""
        new = copy.copy(self)
        new.data = None
        newer = copy.deepcopy(new)
        return newer

    def sort_by_veh_time(self, force: bool = False) -> None:
        """Sort the DwellSet by vehicle and time."""
        if not force:
            logger.info("Checking if DwellSet is sorted by vehicle and time.")
            is_sorted = self.is_sorted_by_veh_time()
            if is_sorted:
                logger.info("DwellSet is already sorted by vehicle and time.")
        if force or not is_sorted:
            logger.info("Sorting DwellSet by vehicle and time.")
            if self.data.index.name in self.get_tracked_cols():
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
        """Sort a dataframe by a group and a time, then set the index."""
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
        """Check if the DwellSet is sorted by start time within each vehicle."""
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
        """Check if the dataset is sorted by a time column within groups.

        We assume here that the grouping column must be the index, since other
        operations in the class depend on this, especially if using Dask as the backend
        for speed.
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
        """Calculate if a dataframe is increasing on sort_col within each group."""
        time_sorted = df.groupby(grp_col)[sort_col].agg(
            lambda ser: ser.is_monotonic_increasing
        )
        return time_sorted

    def accum_masked(
        self,
        keep_mask_col: str,
        accum_cols: str | list[str],
        inplace: bool = False,
    ) -> Self | None:
        """Filter out individual dwells while merging trips together.

        Merging trips together means summing the distances traveled of the trips on
        either side of the eliminated trip.

        Note: This can take a very long time if records for a vehicle cross
        partition boundaries

        Note: This function operates inplace for efficiency.
        """
        if self.verify_sorting:
            self.sort_by_veh_time()

        if inplace:
            new = self
        else:
            new = copy.deepcopy(self)

        if isinstance(accum_cols, str):
            accum_cols = [accum_cols]

        logic_cols = [keep_mask_col, self.reset]
        logic_df = new.data.loc[:, logic_cols]
        logic_df = logic_df.rename(columns={keep_mask_col: "keep", new.reset: "reset"})
        logic_dtypes = new._get_recarray_dtypes(logic_df)
        logic_recs = logic_df.to_records(column_dtypes=logic_dtypes, index=False)

        accum_df = new.data.loc[:, accum_cols]
        accum_renamer = {c: f"{c}_{keep_mask_col}" for c in accum_df.columns}
        accum_df = accum_df.rename(columns=accum_renamer)
        accum_dtypes = new._get_recarray_dtypes(accum_df)
        accum_recs = accum_df.to_records(column_dtypes=accum_dtypes, index=False)

        grp_idxs = new.data.groupby(new.veh).indices
        outs = np.recarray((accum_recs.shape[0],), dtype=accum_recs.dtype)
        for grp, idxs in tqdm(
            grp_idxs.items()
        ):  # Using pandas groupby indices to move over dwell recarray
            outs[idxs] = new._accum_masked_grp_core(
                logics=logic_recs[idxs],
                vals=accum_recs[idxs],
                nvals=len(accum_recs.dtype),
                outs=outs[idxs],
            )
        out_df = pd.DataFrame.from_records(outs, index=new.data.index)

        # # TODO: Ensure that illogical values in accumulated columns get marked
        # outs[~logics["keep"]] = np.nan

        # # TODO: Ensure that boolean reset column gets treated correctly if it was used
        # out_reset_col = accum_renamer[new.reset]
        # out_df[out_reset_col] = out_df[out_reset_col].astype(bool)

        new.data = pd.concat([new.data, out_df], axis=1)

        if inplace:
            return None
        else:
            return new

    def _get_recarray_dtypes(self, df: pd.DataFrame) -> dict[str, str]:
        """Convert a DataFrame to a Record Array using opinionated type conversions.

        Note: Boolean values seem to not convert as bytes, so I will use a small unsigned
        integer instead
        """
        col_dtypes = {}
        for col, dtype in zip(df.columns, df.dtypes):
            if dtype.name == np.bool.__name__:
                col_dtypes.update({col: self._replace_dtypes[dtype.name]})
        return col_dtypes

    @staticmethod
    @njit
    def _accum_masked_grp_core(
        logics: np.recarray,
        vals: np.recarray,
        nvals: int,
        outs: np.recarray,
    ) -> np.recarray:
        nsteps = logics.shape[0]
        if not nsteps == vals.shape[0] == outs.shape[0]:
            raise RuntimeError("The three arrays must have the same length.")

        cum_sums = np.zeros(shape=(1,), dtype=vals.dtype)
        for i in range(nsteps):
            if logics["keep"][i]:
                # If we do reset, then the operation is just a copy of the original
                if not logics["reset"][i]:  # With no reset, we apply accumulation
                    for j in range(nvals):  # get_num_fields(vals):
                        cur_val = vals[i][j]
                        cur_sum = cum_sums[i][j]
                        outs[i][j] = cur_val + cur_sum
                    # outs["reset"][i] = cum_res # Assuming that this will happen automatically with sum
                for j in range(nvals):  # get_num_fields(vals):
                    cum_sums[0][j] = 0
            elif logics["reset"][i]:  # Implicitly, this is reset and not keep
                cum_sums[0] = vals[i]
            else:
                for j in range(nvals):  # get_num_fields(vals):
                    cum_sums[i][j] = vals[i][j] + cum_sums[0][j]
                # cum_res will just reassign to itself, since the current reset is False
        return outs

    def drop_masked(
        self, keep_mask_col: str, inplace: bool = False, drop_mask_col: bool = True
    ) -> Self | None:
        """Drop rows using a boolean mask column."""
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
        """Filter out individual dwells while forcing a reset in the new gaps.

        Note: This function operates inplace for efficiency.
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
        new_name = f"{reset_col}_{keep_mask_col}"
        grp.loc[:, new_name] = DwellSet._reset_masked_grp_core(
            keep=grp[keep_mask_col].values,
            reset=grp[reset_col].values,
        )
        return grp

    @staticmethod
    @njit
    def _reset_masked_grp_core(keep: np.ndarray, reset: np.ndarray) -> np.ndarray:
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
        """Set the default reset_state_before column to assuming that only the first
        dwell for each vehicle requires a reset before.
        """
        self.data[self.reset] = False
        reset_col_idx = self.data.columns.get_loc(self.reset)
        if self.is_dask:
            self.data = self.data.groupby(self.veh, group_keys=False, sort=False).apply(
                DwellSet._set_default_reset_col_grp,
                reset_col_idx=reset_col_idx,
                meta=dd.utils.make_meta(self.data),
            )
        else:
            tqdm.pandas()
            self.data = self.data.groupby(
                self.veh, group_keys=False, sort=False
            ).progress_apply(
                DwellSet._set_default_reset_col_grp,
                reset_col_idx=reset_col_idx,
            )

    @staticmethod
    def _set_default_reset_col_grp(
        grp: pd.DataFrame, reset_col_idx: int
    ) -> pd.DataFrame:
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
        sorted: bool = False,
        *args,
        **kwargs,
    ):
        """Create a DwellSet from trip-formatted data."""
        dw = cls(
            data=trips,
            veh=veh,
            hex=hex,
            start=end_trip,
            end=start_trip,  # This is almost true, since the dwell end is the shifted start_trip
            trip_dist=trip_dist,
            trip_dur=trip_dur,
        )
        if not sorted:
            dw.sort_by_veh_time()
            dw.set_default_reset_col()

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

    def to_events(self) -> dd.DataFrame | pd.DataFrame:
        """Convert dwells into hexagon event profiles."""
        if self.seq_names is None:
            raise RuntimeError("Sequence names must be set before calling 'to_events'.")

        drop_idx = False if self.data.index.name in self.get_tracked_cols() else True
        kws = dict(
            id_cols=[self.veh, self.hex],
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
        """Convert one-dwell-per-row format to one-event-per-row format for a single
        pandas DataFrame.
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

    def to_geodataframe(self, geom_type: str = "point") -> Self:
        """Convert the underlying dataset into a GeoDataFrame."""
        self.data = add_geometries(self.data, hex_col=self.hex, geom_type=geom_type)


def load_dwell_set(dwells: pd.DataFrame, params: dict) -> DwellSet:
    """Load the dwell set from disk with column name parameters."""
    dw = DwellSet(data=dwells, **params)
    return dw


def save_dwell_set(dw: DwellSet) -> pd.DataFrame:
    return dw.data


def get_num_fields(recarr):
    pass


@overload(get_num_fields)
def ol_get_num_fields(recarr):
    nfields = len(recarr.dtype.fields)
    print(f"Type inference time, has {nfields} fields")
    return lambda recarr: nfields
