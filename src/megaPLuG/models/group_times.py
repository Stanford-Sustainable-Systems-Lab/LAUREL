from typing import Self

import pandas as pd
from megaPLuG.utils.time import calc_local_time_attrs, get_local_time_attr_col_name

WEEKEND_FIRST_DAY = 5


class HourOfWeekdayGrouper:
    """Note: This class has only been tested with the frequency of '1h'."""

    freq: str
    time_col: str
    tz_col: str
    count_col: str = "possible_count"

    def __init__(
        self: Self,
        time_col: str,
        tz_col: str,
        freq: str,
    ) -> None:
        self.freq = freq
        self.time_col = time_col
        self.tz_col = tz_col

    @property
    def time_group_cols(self: Self) -> list[str]:
        return ["is_weekend", "time_local_hour"]

    def add_group_classes(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the group classes columns to a dataframe."""
        df = calc_local_time_attrs(
            df=df,
            time_cols=self.time_col,
            attrs=["day_of_week", "hour"],
            tz_col=self.tz_col,
        )
        dow_col = get_local_time_attr_col_name(self.time_col, "day_of_week")
        df["is_weekend"] = df[dow_col] >= WEEKEND_FIRST_DAY
        df = df.drop(columns=[dow_col])
        return df

    def get_possible_obs_counts(self: Self, events: pd.DataFrame) -> pd.DataFrame:
        """Get the counts of possible observations by time group between the first and
        last considered times.
        """
        frame_start = events[self.time_col].min().floor(self.freq)
        frame_end = events[self.time_col].max().ceil(self.freq)
        poss_times = pd.date_range(start=frame_start, end=frame_end, freq=self.freq)
        poss_times = poss_times.to_series(name=self.time_col).reset_index(drop=True)
        all_tz_times = {tz: poss_times for tz in events[self.tz_col].unique()}
        all_tz_times = pd.concat(all_tz_times, names=[self.tz_col])
        all_tz_times = all_tz_times.reset_index().drop(columns=["level_1"])
        all_tz_times[self.tz_col] = pd.Categorical(all_tz_times[self.tz_col])

        all_tz_times = self.add_group_classes(all_tz_times)

        count_grp_cols = [self.tz_col] + self.time_group_cols
        group_counts = all_tz_times.groupby(count_grp_cols, observed=True)[
            self.time_col
        ].count()
        group_counts.name = self.count_col
        return group_counts
