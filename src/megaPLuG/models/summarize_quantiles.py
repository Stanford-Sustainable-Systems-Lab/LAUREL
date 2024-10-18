import logging
from typing import Self

import numpy as np
import pandas as pd
from megaPLuG.utils.time import calc_local_time_attrs, calc_time_zones_from_hexes
from numba import jit
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LoadProfileQuantileSummarizer:
    """Note: This class has only been tested with the frequency of '1h'."""

    time_col: str
    dur_col: str
    power_col: str
    region_col: str
    freq: str
    quantiles: np.ndarray
    # TODO: Separate this out into the subclass
    WEEKEND_FIRST_DAY = 5

    def __init__(
        self: Self,
        time_col: str,
        dur_col: str,
        power_col: str,
        region_col: str,
        freq: str,
        quantiles: np.ndarray,
    ) -> None:
        self.time_col = time_col
        self.dur_col = dur_col
        self.power_col = power_col
        self.region_col = region_col
        self.freq = freq
        self.quantiles = quantiles

    def summarize(self: Self, profiles: pd.DataFrame) -> pd.DataFrame:
        logger.info("Expanding events to cover all groups across their duration")
        # First drop the observations with no duration or zero power
        nonzero = profiles.dropna(subset=[self.dur_col])
        nonzero = nonzero.reset_index()
        drop_idx = nonzero.loc[nonzero[self.power_col] == 0].index
        nonzero = nonzero.drop(index=drop_idx)

        # Then apply expansion
        nonzero_exp = self.expand_events(nonzero)

        logger.info("Grouping events")
        grped_nonzero = self.group_events(nonzero_exp)

        logger.info("Calculating quantiles")
        quantiles = self.calc_sparse_quantiles(grped_nonzero)
        return quantiles

    # TODO: Make this an abstract method as well
    @property
    def time_group_cols(self: Self) -> list[str]:
        return ["is_weekend", "time_local_hour"]

    # TODO: Make this an abstract method so that I can define different grouping methods
    def build_grouper_cols(
        self: Self, times: pd.Series, tzs: pd.Series
    ) -> pd.DataFrame:
        """Build a dataframe with all the columns used to group profile times by, and
        only those columns.
        """
        time_frame = pd.DataFrame({"time": times, "tz": tzs})
        grps = calc_local_time_attrs(
            df=time_frame,
            time_cols="time",
            attrs=["day_of_week", "hour"],
            tz_col="tz",
        )
        grps["is_weekend"] = grps["time_local_day_of_week"] >= self.WEEKEND_FIRST_DAY
        grps = grps.drop(columns=["time", "tz", "time_local_day_of_week"])
        grps = grps.loc[:, self.time_group_cols]
        return grps

    def expand_events(self: Self, events: pd.DataFrame) -> pd.DataFrame:
        """Expand out events to cover all intermediate time units given their start and
        end timestamps.
        """
        end_of_time_group = events[self.time_col].dt.ceil(self.freq)
        end_of_event = events[self.time_col] + events[self.dur_col]
        events["overflow"] = end_of_time_group < end_of_event
        need_expansion = events.loc[events["overflow"]]

        expanded = self._expand_events_wrapper(need_expansion)

        keep_cols = [self.region_col, self.time_col, self.power_col]
        not_expanded = events.loc[~events["overflow"], keep_cols]
        all_nonzero = pd.concat([expanded, not_expanded], axis=0, ignore_index=True)
        return all_nonzero

    def _expand_events_wrapper(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert pandas Series to numpy arrays and set dtypes for self._expand_events_core()."""
        orig_time_type = df[self.time_col].dtype
        starts = df[self.time_col].dt.floor(self.freq)
        ends = (df[self.time_col] + df[self.dur_col]).dt.floor(self.freq)

        out_ids, out_times, out_vals = self._expand_events_core(
            starts=starts.values.astype(np.int64),
            ends=ends.values.astype(np.int64),
            vals=df[self.power_col].values,
            ids=df[self.region_col].values,
            tstep_ns=pd.Timedelta(self.freq).value,
        )

        out_times = pd.to_datetime(out_times.astype("datetime64[ns]"), utc=True)
        out_times = out_times.astype(orig_time_type)
        out = pd.DataFrame(
            {
                self.region_col: out_ids,
                self.time_col: out_times,
                self.power_col: out_vals,
            }
        )
        return out

    @staticmethod
    @jit
    def _expand_events_core(
        starts: np.ndarray[np.int64],
        ends: np.ndarray[np.int64],
        vals: np.ndarray,
        ids: np.ndarray,
        tstep_ns: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand out events to cover all intermediate time units given their start and end
        timestamps.

        To work properly, the starts and ends arguments must be arrays of np.datetime64[h]
        which have been cast to integer.
        """
        n_periods = starts.shape[0]
        ends_plus = ends + tstep_ns
        n_stamps = np.floor_divide(ends_plus - starts, tstep_ns)
        tot_stamps = np.sum(n_stamps)
        times_exp = np.zeros(tot_stamps, dtype=starts.dtype)
        vals_exp = np.zeros(tot_stamps, dtype=vals.dtype)
        ids_exp = np.zeros(tot_stamps, dtype=ids.dtype)
        cursor = 0
        for i in range(n_periods):
            next_cursor = cursor + n_stamps[i]
            idxs = np.arange(cursor, next_cursor, dtype=np.int64)
            times_exp[idxs] = np.arange(
                starts[i], ends_plus[i], tstep_ns, dtype=times_exp.dtype
            )
            vals_exp[idxs] = vals[i]
            ids_exp[idxs] = ids[i]
            cursor = next_cursor
        return ids_exp, times_exp, vals_exp

    def group_events(self: Self, events: pd.DataFrame) -> pd.DataFrame:
        """Group events by the groups defined in `build_grouper_cols`."""
        grouper = [self.region_col, pd.Grouper(key=self.time_col, freq=self.freq)]
        events = events.groupby(grouper)[self.power_col].max()
        events = events.reset_index()

        events = calc_time_zones_from_hexes(
            df=events,
            hex_col="hex_id",  # TODO: Regions other than the hexagon will fail here. I could replace this by a merge or a requirement that the profiles have timezones on them.
        )
        grouper_cols = self.build_grouper_cols(
            times=events[self.time_col],
            tzs=events["tz"],
        )
        events = pd.concat([events, grouper_cols], axis=1)
        return events

    def calc_sparse_quantiles(self: Self, events: pd.DataFrame) -> pd.DataFrame:
        """Calculate quantiles using observations paired with the count of possible
        observations to represent zeros.
        """
        group_counts = self._get_possible_obs_counts(events)
        events = events.merge(group_counts, how="left", on=group_counts.index.names)
        group_cols = [self.region_col] + self.time_group_cols
        grouping = events.groupby(group_cols)
        grp_idxs = grouping.indices
        powers = events[self.power_col].values
        possibles = events["possible_times"].values
        results = np.zeros(
            (grouping.ngroups, self.quantiles.shape[0]), dtype=np.float64
        )

        i = 0
        for _, idx in tqdm(grp_idxs.items()):
            cur_possibles = possibles[idx[0]]
            cur_vals = powers[idx]
            results[i, :] = self._calc_sparse_quantiles_core(
                n_obs=cur_possibles,
                nonzeros=cur_vals,
                quantiles=self.quantiles,
            )
            i += 1
        quantile_df = pd.DataFrame(
            data=results, index=grp_idxs.keys(), columns=self.quantiles
        )
        quantile_df.index.names = group_cols
        return quantile_df

    @staticmethod
    @jit
    def _calc_sparse_quantiles_core(
        n_obs: int,
        nonzeros: np.ndarray,
        quantiles: np.ndarray[np.float64],
    ) -> np.ndarray:
        arr = np.zeros(n_obs, dtype=nonzeros.dtype)
        n_nonzero = nonzeros.shape[0]
        arr[-n_nonzero:] = nonzeros
        qtls = np.quantile(a=arr, q=quantiles)
        return qtls

    def _get_possible_obs_counts(self: Self, events: pd.DataFrame) -> pd.DataFrame:
        """Get the counts of possible observations by time group between the first and
        last considered times.
        """
        frame_start = events[self.time_col].min().floor(self.freq)
        frame_end = events[self.time_col].max().ceil(self.freq)
        poss_times = pd.date_range(start=frame_start, end=frame_end, freq=self.freq)
        poss_times = poss_times.to_series(name="possible_times").reset_index(drop=True)
        all_tz_times = {tz: poss_times for tz in events["tz"].cat.categories}
        all_tz_times = pd.concat(all_tz_times, names=["tz"])
        all_tz_times = all_tz_times.reset_index().drop(columns=["level_1"])
        all_tz_times["tz"] = pd.Categorical(all_tz_times["tz"])

        grpers = self.build_grouper_cols(
            times=all_tz_times["possible_times"],
            tzs=all_tz_times["tz"],
        )
        all_tz_times = pd.concat([all_tz_times, grpers], axis=1)

        count_grp_cols = ["tz"] + self.time_group_cols
        group_counts = all_tz_times.groupby(count_grp_cols, observed=True).count()
        return group_counts
