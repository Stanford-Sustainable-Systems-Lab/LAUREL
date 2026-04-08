"""Time-grouping classes for aggregating load profiles and normalising observation counts.

Provides :class:`AbstractTimeGrouper` and three concrete implementations used
to group dwell events and count how many observations are possible within each
time group across the study period.  The "possible observation count" is used
by :class:`~laurel.models.summarize.NonzeroGroupedSummarizer` to correctly
include zeros in quantile calculations.

Concrete groupers
-----------------
- :class:`HourOfWeekdayGrouper`: groups by (is_weekend × local hour of day).
- :class:`LocalHourOfDayGrouper`: groups by local hour of day only.
- :class:`AdaptiveTimeGrouper`: infers the minimum set of time attributes
  needed to uniquely identify each bin within the study window (e.g. year,
  day-of-week, hour), adapting to arbitrary ``freq`` strings.

Key design decisions
--------------------
- **Possible count computation**: :meth:`AbstractTimeGrouper.get_possible_obs_counts`
  builds a full UTC time range between the study start and end, converts to
  each possible timezone in ``possible_tzs``, applies ``add_group_classes``,
  and counts how many UTC timestamps fall in each (timezone × group) bin.
  This correctly handles DST transitions and unequal bin sizes.
- **Timezone handling**: all internal timestamps are UTC; local-time conversion
  is applied only to compute group labels, not stored in the data.  This avoids
  ambiguous or non-existent local times at DST boundaries.
- **``WEEKEND_FIRST_DAY = 5``**: ISO weekday numbering (Monday = 0); day 5 is
  Saturday, so ``day_of_week >= 5`` selects Saturday and Sunday.
"""

from abc import ABC, abstractmethod
from typing import Self

import pandas as pd

from laurel.utils.time import calc_local_time, calc_time_attrs

WEEKEND_FIRST_DAY = 5


class AbstractTimeGrouper(ABC):
    """Abstract base class for time-grouping strategies.

    Subclasses implement :meth:`add_group_classes` to assign each timestamp to
    a group label, and declare :attr:`time_group_cols` and :attr:`time_attrs`
    to describe the resulting grouping columns.

    Class attributes:
        count_col: Name of the output column produced by
            :meth:`get_possible_obs_counts`.  Defaults to ``"possible_count"``.
        _default_tz: Fallback timezone string (``"America/Los_Angeles"``).
    """

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
        """Initialise the time grouper with study window and timezone parameters.

        Args:
            time_col: Name of the timestamp column in DataFrames passed to
                :meth:`add_group_classes`.
            tz_col: Name of the column holding IANA timezone strings.
            start_time: Start of the study period.  Floored to ``freq``.
            end_time: End of the study period.  Ceiled to ``freq``.
            possible_tzs: Complete list of IANA timezone strings that appear
                in the data.  All are used in
                :meth:`get_possible_obs_counts`.
            freq: Pandas frequency string for the time bins (default ``"1h"``).
        """
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
        """Build a Series of regularly-spaced timestamps covering the study window.

        Args:
            **kwargs: Extra keyword arguments forwarded to ``pd.date_range``
                (e.g. ``tz="UTC"``).

        Returns:
            Series named :attr:`time_col` with integer RangeIndex.
        """
        poss_times = pd.date_range(
            start=self.start_time, end=self.end_time, freq=self.freq, **kwargs
        )
        poss_times = poss_times.to_series(name=self.time_col).reset_index(drop=True)
        return poss_times

    def get_possible_obs_counts(self: Self) -> pd.DataFrame:
        """Count how many UTC timestamps fall in each (timezone, group) bin.

        Builds a full UTC time range from :attr:`start_time` to
        :attr:`end_time`, replicates it for every timezone in
        :attr:`possible_tzs`, converts to local time, applies
        :meth:`add_group_classes`, and counts occurrences per
        ``(tz_col, *time_group_cols)`` combination.

        Returns:
            Series named :attr:`count_col`, indexed by
            ``(tz_col, *time_group_cols)``, giving the number of UTC
            timestamps in each bin.
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
    """Group timestamps by (is_weekend × local hour of day).

    Produces two grouping columns: ``is_weekend`` (``True`` for Saturday and
    Sunday) and ``time_local_hour`` (0–23).  Only tested with ``freq="1h"``.
    """

    _time_attrs: list[str] = ["day_of_week", "hour"]
    _time_group_cols: list[str] = ["is_weekend", "time_local_hour"]

    @property
    def time_group_cols(self: Self) -> list[str]:
        return self._time_group_cols

    @property
    def time_attrs(self: Self) -> list[str]:
        return self._time_attrs

    def add_group_classes(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``is_weekend`` and ``time_local_hour`` columns to ``df``.

        Converts :attr:`time_col` to local time using :attr:`tz_col`, extracts
        ``day_of_week`` and ``hour``, derives ``is_weekend`` as
        ``day_of_week >= WEEKEND_FIRST_DAY``, and drops the intermediate
        ``day_of_week`` column.

        Args:
            df: DataFrame containing :attr:`time_col` and :attr:`tz_col`.

        Returns:
            ``df`` with ``is_weekend`` and ``time_local_hour`` columns added.
        """
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
    """Group timestamps by local hour of day only (ignoring weekday).

    Produces a single grouping column ``slice_time_relative`` (0–23).
    Only tested with ``freq="1h"``.
    """

    _time_attrs: list[str] = ["hour"]
    _time_group_cols: list[str] = ["slice_time_relative"]

    @property
    def time_group_cols(self: Self) -> list[str]:
        return self._time_group_cols

    @property
    def time_attrs(self: Self) -> list[str]:
        return self._time_attrs

    def add_group_classes(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the ``slice_time_relative`` (hour) column to ``df``.

        Args:
            df: DataFrame containing :attr:`time_col`.

        Returns:
            ``df`` with ``slice_time_relative`` column added.
        """
        df = calc_time_attrs(df=df, time_col=self.time_col, attrs=self.time_attrs)
        return df


class AdaptiveTimeGrouper(AbstractTimeGrouper):
    """Group timestamps using the minimum set of attributes needed for the study window.

    At construction time, inspects a sample ``pd.date_range`` of the study
    period at the given ``freq`` to determine which time components (year,
    day-of-week, hour, minute, second) actually vary — only those are used as
    grouping attributes.  This correctly handles both sub-hourly and
    multi-day study periods without configuration.

    Only tested with ``freq="1h"``.
    """

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
        """Set the list of time attributes used to define group classes."""
        self._time_attrs = attrs

    def add_group_classes(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the adaptively-determined time-attribute columns to ``df``.

        Args:
            df: DataFrame containing :attr:`time_col`.

        Returns:
            ``df`` with one column per entry in :attr:`time_attrs` added.
        """
        df = calc_time_attrs(df=df, time_col=self.time_col, attrs=self.time_attrs)
        return df

    def _get_time_attrs_from_freq(self: Self) -> list[str]:
        """Identify which time components vary across the study window at ``freq``.

        Builds a ``pd.date_range`` from :attr:`start_time` to :attr:`end_time`
        at :attr:`freq`, then checks each attribute in ``_test_time_attrs``
        to see whether its value changes relative to the first timestamp.
        Only varying attributes are included.

        Returns:
            Ordered list of time-attribute names (subset of
            ``_test_time_attrs``) needed to uniquely identify each bin.
        """
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
