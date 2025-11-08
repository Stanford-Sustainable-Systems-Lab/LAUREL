from abc import ABC, abstractmethod
from typing import Self

import pandas as pd

from megaplug.utils.time import calc_local_time, calc_time_attrs

WEEKEND_FIRST_DAY = 5


class AbstractTimeGrouper(ABC):
    """Abstract class for creating time groupings and evaluating their size."""

    freq: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    possible_tzs: list[str]
    time_col: str
    tz_col: str
    count_col: str = "possible_count"
    _time_attrs: list[str] = []
    _time_group_cols: list[str] = []
    _default_tz: str = "America/Los_Angeles"

    def __init__(
        self: Self,
        time_col: str,
        tz_col: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        possible_tzs: list[str],
        freq: str = "1h",
    ) -> None:
        self.time_col = time_col
        self.tz_col = tz_col
        self.freq = freq
        self.start_time = start_time.floor(self.freq)
        self.end_time = end_time.ceil(self.freq)
        self.possible_tzs = possible_tzs

    @property
    @abstractmethod
    def time_group_cols(self: Self) -> list[str]:
        return self._time_group_cols

    @property
    @abstractmethod
    def time_attrs(self: Self) -> list[str]:
        return self._time_attrs

    @abstractmethod
    def add_group_classes(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the group classes columns to a dataframe."""
        pass

    def _build_time_range(self: Self, **kwargs: dict) -> pd.Series:
        """Build a series of the time range of times considered."""
        poss_times = pd.date_range(
            start=self.start_time, end=self.end_time, freq=self.freq, **kwargs
        )
        poss_times = poss_times.to_series(name=self.time_col).reset_index(drop=True)
        return poss_times

    def get_possible_obs_counts(self: Self) -> pd.DataFrame:
        """Get the counts of possible observations by time group between the first and
        last considered times.
        """
        poss_times = self._build_time_range(tz="UTC")
        all_tz_times = {tz: poss_times for tz in self.possible_tzs}
        all_tz_times = pd.concat(all_tz_times, names=[self.tz_col])
        all_tz_times = all_tz_times.reset_index().drop(columns=["level_1"])
        all_tz_times[self.tz_col] = pd.Categorical(all_tz_times[self.tz_col])

        all_tz_times = self.add_group_classes(all_tz_times)

        count_grp_cols = [self.tz_col] + self.time_group_cols
        group_counts = all_tz_times.value_counts(subset=count_grp_cols).sort_index()
        group_counts.name = self.count_col
        return group_counts


class HourOfWeekdayGrouper(AbstractTimeGrouper):
    """Note: This class has only been tested with the frequency of '1h'."""

    _time_attrs: list[str] = ["day_of_week", "hour"]
    _time_group_cols: list[str] = ["is_weekend", "time_local_hour"]

    @property
    def time_group_cols(self: Self) -> list[str]:
        return self._time_group_cols

    @property
    def time_attrs(self: Self) -> list[str]:
        return self._time_attrs

    def add_group_classes(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the group classes columns to a dataframe."""
        local_col = self.time_col + "_local"
        df = calc_local_time(
            df=df,
            time_cols=self.time_col,
            local_cols=local_col,
            tz_col=self.tz_col,
        )
        df = calc_time_attrs(df=df, time_col=local_col, attrs=self.time_attrs)
        dow_col = f"{local_col}_day_of_week"
        df["is_weekend"] = df[dow_col] >= WEEKEND_FIRST_DAY
        df = df.drop(columns=[dow_col])
        return df


class LocalHourOfDayGrouper(AbstractTimeGrouper):
    """Note: This class has only been tested with the frequency of '1h'."""

    _time_attrs: list[str] = ["hour"]
    _time_group_cols: list[str] = ["slice_time_relative"]

    @property
    def time_group_cols(self: Self) -> list[str]:
        return self._time_group_cols

    @property
    def time_attrs(self: Self) -> list[str]:
        return self._time_attrs

    def add_group_classes(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the group classes columns to a dataframe."""
        df = calc_time_attrs(df=df, time_col=self.time_col, attrs=self.time_attrs)
        return df


class AdaptiveTimeGrouper(AbstractTimeGrouper):
    """Note: This class has only been tested with the frequency of '1h'."""

    _time_attrs: list[str] = []
    _time_group_cols: list[str] = ["slice_time_relative"]
    _test_time_attrs: list[str] = ["year", "day_of_week", "hour", "minute", "second"]

    def __init__(self, time_col, tz_col, start_time, end_time, possible_tzs, freq: str):
        super().__init__(
            time_col, tz_col, start_time, end_time, possible_tzs, freq=freq
        )
        self.time_attrs = self._get_time_attrs_from_freq()

    @property
    def time_group_cols(self: Self) -> list[str]:
        return self._time_group_cols

    @property
    def time_attrs(self: Self) -> list[str]:
        return self._time_attrs

    @time_attrs.setter
    def time_attrs(self: Self, attrs: list[str]) -> None:
        """Set the time attributes to use for grouping."""
        self._time_attrs = attrs

    def add_group_classes(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the group classes columns to a dataframe."""
        df = calc_time_attrs(df=df, time_col=self.time_col, attrs=self.time_attrs)
        return df

    def _get_time_attrs_from_freq(self: Self) -> list[str]:
        """Get the time attributes required by the given frequency and time window."""
        test_range = pd.date_range(
            start=self.start_time,
            end=self.end_time,
            freq=self.freq,
        )

        components = []
        for att in self._test_time_attrs:
            att_vals = test_range.__getattribute__(att)
            base_att_val = test_range[0].__getattribute__(att)
            if (att_vals != base_att_val).any():
                components.append(att)
        return components
