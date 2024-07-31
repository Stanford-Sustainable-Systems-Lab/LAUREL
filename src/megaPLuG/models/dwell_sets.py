import re
from itertools import product

import dask.dataframe as dd
import numpy as np
import pandas as pd


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
    _dist = None
    _reset = None
    data = None
    data_type = None

    def __init__(
        self,
        data: dd.DataFrame | pd.DataFrame,
        veh: str,
        hex: str,
        start: str,
        end: str,
        dist: str,
        reset: str = None,
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

        dup_test = [veh, hex, start, end, dist]
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
        self._dist = _return_if_present(dist)
        if reset is None:
            self._reset = "reset_state_before"
            self.set_default_reset_col()
        else:
            self._reset = _return_if_present(reset)
        self.standardize_names()
        self.seq_names = [
            "dwell_start",
            "dwell_end",
        ]  # Should be used for the sequence of events

    def sort_by_veh_time(self) -> None:
        """Sort the DwellSet by vehicle and time."""
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
        df = df.sort_values(
            [grp_col, time_col]
        )  # Test if this works if grp_col is already the index
        if df.index.name != grp_col:
            df = df.reset_index(drop=drop_cur_idx)
            if isinstance(df, dd.DataFrame):
                df = df.set_index(grp_col, sorted=True)
            elif isinstance(df, pd.DataFrame):
                df = df.set_index(grp_col)
        return df

    def filter_through(self, keep_col: str):
        """Filter out individual dwells while merging trips together.

        Merging trips together means summing the distances traveled of the trips on
        either side of the eliminated trip.

        Note: This can take a very long time if records for a vehicle cross
        partition boundaries
        """
        if isinstance(self.data, dd.DataFrame):
            self.data = self.data.groupby(self.veh, group_keys=False).apply(
                DwellSet._filter_through_grp,
                keep_col=keep_col,
                reset_col=self.reset,
                dist_col=self.dist,
                meta=dd.utils.make_meta(self.data),
            )
        elif isinstance(self.data, pd.DataFrame):
            self.data = self.data.groupby(self.veh, group_keys=False).apply(
                DwellSet._filter_through_grp,
                keep_col=keep_col,
                reset_col=self.reset,
                dist_col=self.dist,
            )
        self.data[keep_col] = self.data[keep_col].replace(False, np.NaN)
        self.data = self.data.dropna(subset=keep_col)
        self.data = self.data.drop(columns=keep_col)

    @staticmethod
    def _filter_through_grp(
        grp: pd.DataFrame, keep_col: str, reset_col: str, dist_col: str
    ) -> pd.DataFrame:
        dist_cum = grp[dist_col].cumsum()
        reset_cum = grp[reset_col].cumsum()
        mask = np.logical_or(grp[keep_col], grp[reset_col].shift(-1, fill_value=False))

        def _filter_through_col(cum: pd.Series, mask: np.ndarray) -> np.ndarray:
            base = cum * mask
            base = base / mask
            base = base.ffill().shift(1, fill_value=0)
            return cum - base

        grp[dist_col] = _filter_through_col(dist_cum, mask)
        grp[reset_col] = _filter_through_col(reset_cum, mask).astype(bool)
        return grp

    def filter_reset(self, keep_col: str):
        """Filter out individual dwells while forcing a reset in the new gaps."""
        if isinstance(self.data, dd.DataFrame):
            self.data = self.data.groupby(self.veh, group_keys=False).apply(
                DwellSet._filter_reset_grp,
                keep_col=keep_col,
                reset_col=self.reset,
                meta=dd.utils.make_meta(self.data),
            )
        elif isinstance(self.data, pd.DataFrame):
            self.data = self.data.groupby(self.veh, group_keys=False).apply(
                DwellSet._filter_reset_grp,
                keep_col=keep_col,
                reset_col=self.reset,
            )
        self.data[keep_col] = self.data[keep_col].replace(False, np.NaN)
        self.data = self.data.dropna(subset=keep_col)
        self.data = self.data.drop(columns=keep_col)

    @staticmethod
    def _filter_reset_grp(
        grp: pd.DataFrame, keep_col: str, reset_col: str
    ) -> pd.DataFrame:
        keep_diff = (
            grp[keep_col].astype(float).diff().fillna(1)
        )  # Perhaps could get speedup here by removing casting, but I need to think more about validity.
        keep_diff = np.maximum(0, keep_diff).astype(bool)
        grp[reset_col] = np.logical_or(keep_diff, grp[reset_col])
        return grp

    def set_default_reset_col(self):
        """Set the default reset_state_before column to assuming that only the first
        dwell for each vehicle requires a reset before.
        """
        self.data[self.reset] = False
        if isinstance(self.data, dd.DataFrame):
            self.data = self.data.groupby(self.veh, group_keys=False).apply(
                DwellSet._set_default_reset_col_grp,
                reset_col=self.reset,
                meta=dd.utils.make_meta(self.data),
            )
        elif isinstance(self.data, pd.DataFrame):
            self.data = self.data.groupby(self.veh, group_keys=False).apply(
                DwellSet._set_default_reset_col_grp,
                reset_col=self.reset,
            )

    @staticmethod
    def _set_default_reset_col_grp(grp: pd.DataFrame, reset_col: str) -> pd.DataFrame:
        grp.loc[grp.index[0], reset_col] = True
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

    def standardize_names(self):
        """Standardize names of the tracked data columns."""
        self.veh = "veh_id"
        self.start = "dwell_start_utc"
        self.end = "dwell_end_utc"
        self.hex = "hex_id"
        self.dist = "dist_arriving"
        self.reset = "reset_state_before"

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
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, value):
        self._rename_idx_col(value, self._dist)
        self._dist = value

    @property
    def reset(self):
        return self._reset

    @reset.setter
    def reset(self, value):
        self._rename_idx_col(value, self._reset)
        self._reset = value

    def _rename_idx_col(self, new: str, old: str):
        if self.data.index.name == old:
            self.data.index = self.data.index.rename(new)
        else:
            self.data = self.data.rename(columns={old: new})

    def get_tracked_cols(self):
        return [self._veh, self._hex, self._start, self._end, self._dist, self._reset]

    @classmethod
    def from_trips(
        cls,
        trips: dd.DataFrame,
        veh: str,
        hex: str,
        start_trip: str,
        end_trip: str,
        dist: str,
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
            dist=dist,
        )
        dw.standardize_names()
        if not sorted:
            dw.sort_by_veh_time()

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
        return dw

    def to_hex_profiles(self) -> dd.DataFrame | pd.DataFrame:
        """Convert dwells into hexagon event profiles."""
        if self.data.index.name in self.get_tracked_cols():
            drop = False
        else:
            drop = True

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
        tcol = DwellSet._get_seq_name_tail(self.seq_names[0], self.start)
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
        return re.findall(f"(?<={seq_name}_).+", col_name)[0]
