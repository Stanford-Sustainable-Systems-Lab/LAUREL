import copy
import logging
import re
from collections.abc import Callable
from itertools import product
from typing import Self

import dask.dataframe as dd
import dask_geopandas
import geopandas as gpd
import numpy as np
import pandas as pd
from megaPLuG.utils.h3 import cells_to_points, cells_to_polygons
from numba import njit
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
    data = None
    data_type = None

    def __init__(
        self,
        data: dd.DataFrame | pd.DataFrame,
        veh: str,
        hex: str,
        start: str,
        end: str,
        trip_dist: str,
        trip_dur: str,
        reset: str = None,
        verify_sorting: bool = True,
        *args,
        **kwargs,
    ):
        if isinstance(data, dd.DataFrame):
            self.data_type = dd.DataFrame
        elif isinstance(data, pd.DataFrame):
            self.data_type == pd.DataFrame
        else:
            raise RuntimeError("Unsupported underlying data type for DwellSet.")
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

    def filter_through(
        self,
        keep_mask_col: str,
        sum_cols: str | list[str] | None = None,
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

        if sum_cols is None:
            sum_cols = self.trip_dist
        if isinstance(sum_cols, str):
            sum_cols = [sum_cols]

        sums_master_dtype = np.result_type(*self.data[sum_cols].dtypes.values)
        for col in sum_cols:
            self.data[col] = self.data[col].astype(sums_master_dtype)

        # Force numba compilation
        base = np.array(
            [True, False, True]
        )  # Just an example array of the correct dtype
        sums_base = np.expand_dims(base.astype(sums_master_dtype), axis=1)
        if len(sum_cols) > 1:
            sums_base = np.hstack([sums_base] * len(sum_cols))
        _ = DwellSet._filter_through_grp_core(
            keep=base,
            sums=sums_base,
            reset=base,
        )
        if inplace:
            new = self
        else:
            new = self.copy_without_data()

        if isinstance(self.data, dd.DataFrame):
            new.data = self.data.groupby(self.veh, group_keys=False).apply(
                DwellSet._filter_through_grp,
                keep_mask_col=keep_mask_col,
                sum_cols=sum_cols,
                reset_col=self.reset,
                meta=dd.utils.make_meta(self.data),
            )
            new.data[keep_mask_col] = new.data[keep_mask_col].replace(False, np.NaN)
            new.data.dropna(subset=keep_mask_col)
            new.data.drop(columns=keep_mask_col)
        elif isinstance(self.data, pd.DataFrame):
            tqdm.pandas()
            new.data = self.data.groupby(self.veh, group_keys=False).progress_apply(
                DwellSet._filter_through_grp,
                keep_mask_col=keep_mask_col,
                sum_cols=sum_cols,
                reset_col=self.reset,
            )
            new.data[keep_mask_col] = new.data[keep_mask_col].replace(False, np.NaN)
            new.data.dropna(subset=keep_mask_col, inplace=True)
            new.data.drop(columns=keep_mask_col, inplace=True)
        if inplace:
            return None
        else:
            return new

    @staticmethod
    def _filter_through_grp(
        grp: pd.DataFrame,
        keep_mask_col: str,
        sum_cols: str | list[str],
        reset_col: str,
    ) -> pd.DataFrame:
        if isinstance(sum_cols, str):
            sum_cols = [sum_cols]
            sums = np.expand_dims(grp[sum_cols].values, axis=1)
        else:
            sums = grp[sum_cols].values

        arr = DwellSet._filter_through_grp_core(
            keep=grp[keep_mask_col].values,
            sums=sums,
            reset=grp[reset_col].values,
        )

        for i, col in enumerate(sum_cols):
            grp.loc[:, col] = arr[:, i]
        grp.loc[:, reset_col] = arr[:, -1].astype(bool)
        return grp

    @staticmethod
    @njit
    def _filter_through_grp_core(
        keep: np.ndarray,
        sums: np.ndarray,
        reset: np.ndarray,
    ) -> pd.DataFrame:
        nsteps = keep.shape[0]
        if not nsteps == reset.shape[0] == sums.shape[0]:
            raise RuntimeError("The three arrays must have the same length.")
        arr = np.hstack((sums, np.expand_dims(reset, axis=1)))

        cum_sums = np.zeros(sums.shape[1], dtype=sums.dtype)
        cum_res = False
        for i in range(nsteps):
            if keep[i]:
                # If we do reset, then the operation is just a copy of the original
                if not reset[i]:  # With no reset, we apply accumulation
                    arr[i, :-1] = sums[i, :] + cum_sums
                    arr[i, -1] = cum_res
                cum_sums[:] = 0
                cum_res = False
            elif reset[i]:  # Implicitly, this is reset and not keep
                cum_sums = sums[i, :]
                cum_res = True
            else:
                cum_sums = sums[i, :] + cum_sums
                # cum_res will just reassign to itself, since the current reset is False
        return arr

    def filter_reset(self, keep_mask_col: str, inplace: bool = False) -> Self | None:
        """Filter out individual dwells while forcing a reset in the new gaps.

        Note: This function operates inplace for efficiency.
        """
        if self.verify_sorting:
            self.sort_by_veh_time()

        # Force numba compilation
        _ = DwellSet._filter_reset_grp(
            grp=copy.deepcopy(self.data.head(5)),  # copy to prevent double-processing
            keep_mask_col=keep_mask_col,
            reset_col=self.reset,
        )

        if inplace:
            new = self
        else:
            new = self.copy_without_data()

        if isinstance(self.data, dd.DataFrame):
            new.data = self.data.groupby(self.veh, group_keys=False).apply(
                DwellSet._filter_reset_grp,
                keep_mask_col=keep_mask_col,
                reset_col=self.reset,
                meta=dd.utils.make_meta(self.data),
            )
            new.data[keep_mask_col] = new.data[keep_mask_col].replace(False, np.NaN)
            new.data.dropna(subset=keep_mask_col)
            new.data.drop(columns=keep_mask_col)
        elif isinstance(self.data, pd.DataFrame):
            tqdm.pandas()
            new.data = self.data.groupby(self.veh, group_keys=False).progress_apply(
                DwellSet._filter_reset_grp,
                keep_mask_col=keep_mask_col,
                reset_col=self.reset,
            )
            new.data[keep_mask_col] = new.data[keep_mask_col].replace(False, np.NaN)
            new.data.dropna(subset=keep_mask_col, inplace=True)
            new.data.drop(columns=keep_mask_col, inplace=True)
        if inplace:
            return None
        else:
            return new

    @staticmethod
    def _filter_reset_grp(
        grp: pd.DataFrame, keep_mask_col: str, reset_col: str
    ) -> pd.DataFrame:
        grp.loc[:, reset_col] = DwellSet._filter_reset_grp_core(
            keep=grp[keep_mask_col].values,
            reset=grp[reset_col].values,
        )
        return grp

    @staticmethod
    @njit
    def _filter_reset_grp_core(keep: np.ndarray, reset: np.ndarray) -> np.ndarray:
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
        if isinstance(self.data, dd.DataFrame):
            self.data = self.data.groupby(self.veh, group_keys=False).apply(
                DwellSet._set_default_reset_col_grp,
                reset_col_idx=reset_col_idx,
                meta=dd.utils.make_meta(self.data),
            )
        elif isinstance(self.data, pd.DataFrame):
            tqdm.pandas()
            self.data = self.data.groupby(self.veh, group_keys=False).progress_apply(
                DwellSet._set_default_reset_col_grp,
                reset_col_idx=reset_col_idx,
            )

    @staticmethod
    def _set_default_reset_col_grp(
        grp: pd.DataFrame, reset_col_idx: int
    ) -> pd.DataFrame:
        grp.iat[0, reset_col_idx] = True
        return grp

    def is_sorted(self) -> bool:
        """Check if the DwellSet is sorted by vehicle and time.

        This is a precondition for many of the algorithms.
        """
        if self.data.index.name != self._veh_time:
            return False
        is_inc = self.data.groupby(self.veh).apply(
            lambda grp: grp.index.is_monotonic_increasing, include_groups=False
        )
        if isinstance(self.data, dd.DataFrame):
            is_inc = is_inc.compute()
        all_inc = np.all(is_inc)
        if all_inc:
            return True
        else:
            return False

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

        if isinstance(dw.data, dd.DataFrame):
            dw.data = dw.data.groupby(dw.veh, group_keys=False).apply(
                _shift_by_grp,
                src=dw.end,
                tgt=dw.end,
                meta=dd.utils.make_meta(dw.data),
            )
        elif isinstance(dw.data, pd.DataFrame):
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

    def to_hex_profiles(self) -> dd.DataFrame | pd.DataFrame:
        """Convert dwells into hexagon event profiles."""
        if self.data.index.name in self.get_tracked_cols():
            drop = False
        else:
            drop = True

        if self.seq_names is None:
            raise RuntimeError(
                "Sequence names must be set before calling 'to_hex_profiles'."
            )
        tcol = DwellSet._get_seq_name_tail(self.seq_names[0], self.start)

        id_cols = [self.veh, self.hex]
        if isinstance(self.data, dd.DataFrame):
            events = self.data.map_partitions(
                DwellSet._dwells_to_events_grp,
                id_cols=id_cols,
                seq_names=self.seq_names,
                drop_cur_idx=drop,
            )
        elif isinstance(self.data, pd.DataFrame):
            events = DwellSet._dwells_to_events_grp(
                dw=self.data,
                id_cols=id_cols,
                seq_names=self.seq_names,
                drop_cur_idx=drop,
            )

        # Sort by hexagon and time
        events = DwellSet._sort_by_grp_time(
            df=events, grp_col=self.hex, time_col=tcol, drop_cur_idx=True
        )
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

    def to_geodataframe(self, geom_type: str = "point"):
        """Convert the underlying dataset into a GeoDataFrame."""
        if geom_type == "point":
            f = cells_to_points
        elif geom_type == "polygon":
            f = cells_to_polygons
        else:
            raise RuntimeError("Only 'point' and 'polygon' geometries are supported.")

        if isinstance(self.data, dd.DataFrame):
            self.data = dask_geopandas.from_dask_dataframe(df=self.data, geometry=None)
            self.data = self.data.map_partitions(
                DwellSet._cells_to_geom_wrapper,
                f=f,
                hex_col=self.hex,
            )
        elif isinstance(self.data, pd.DataFrame):
            self.data = gpd.GeoDataFrame(data=self.data, geometry=None)
            self.data = DwellSet._cells_to_geom_wrapper(
                gdf=self.data,
                f=f,
                hex_col=self.hex,
            )
        else:
            raise RuntimeError("Only pandas and dask dataframes are supported.")

    @staticmethod
    def _cells_to_geom_wrapper(
        gdf: gpd.GeoDataFrame, f: Callable[[pd.Series], gpd.GeoSeries], hex_col: str
    ) -> pd.DataFrame:
        """Convert a Pandas DataFrame to a GeoDataFrame using its hexagon column."""
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise RuntimeError("Incoming data is not a GeoDataFrame")

        if hex_col in gdf.columns:
            hexes = gdf[hex_col]
        elif hex_col in gdf.index.names:
            hexes = gdf.index.get_level_values(hex_col).to_series()
        else:
            raise RuntimeError(f"'{hex_col}' not found in DataFrame columns or index.")

        geoms = f(hexes)
        gdf = gdf.set_geometry(geoms)
        return gdf


def load_dwell_set(dwells: pd.DataFrame, params: dict) -> DwellSet:
    """Load the dwell set from disk with column name parameters."""
    dw = DwellSet(data=dwells, **params)
    return dw


def save_dwell_set(dw: DwellSet) -> pd.DataFrame:
    return dw.data
