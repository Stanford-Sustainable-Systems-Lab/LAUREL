"""
Utility functions
by Fletcher Passow
February 2023
"""

from collections.abc import Callable

import numpy as np
import pandas as pd


def get_events(
    obs_ser: pd.Series,
    time_ser: pd.Series,
    grp_ser: pd.Series,
    obs_in_event: Callable = (lambda x: x > 0),
    max_time_elapsed: int = 6 * 60,
) -> np.ndarray:
    """Get a ndarray which gives the indices for events based on periodic
    observations.

    Events are defined as semi-contiguous stretches of a value matching the
    criterion given by obs_in_event. The breaks in an event can only be as long as
    max_time_elapsed.

    Args:
      obs_ser: periodic observations to base groups off of
      time_ser: time (as pandas timestamps) when observations in obs_ser were taken
      grp_ser: marker for groups of observations, usually this would be a series
        of integer geyser ids, but if None, then all observations are assumed to
        come from the same group.
      obs_in_event: function taking a series and returning a boolean series which is True for
        every row (observation) that should be in an event and False for every row (observation)
        that should NOT be in an event.
      max_time_elapsed: integer giving the maximum time in seconds between events for them
              to be combined into a single event.

    Returns: a numpy ndarray which gives an index of which event a particular
    observation is a part of (or zero if the observation is part of no event).
    Zero is used instead of a null value to mark the non-events because it allows
    the numpy ndarray to be cast to an integer type, which facilitates merging.
    """
    if grp_ser is None:
        grp_ser = pd.Series(0, index=obs_ser.index)

    event_df = pd.DataFrame(
        {"grp": grp_ser.values, "time": time_ser.values, "obs": obs_ser.values}
    )
    event_df = event_df.set_index(["grp", "time"])
    if not event_df.index.is_monotonic_increasing:
        raise Exception(
            "The given combination of groups and times is not monotonic increasing."
        )

    def get_events_geyser(meas_ser, obs_in_event, max_time_elapsed):
        """Get a series which gives the indices for events for a single geyser."""

        if meas_ser.index.get_level_values("grp").nunique() > 1:
            raise Exception("meas_ser contains more than one grouping.")

        def obs_out_event(x):
            return np.logical_not(obs_in_event(x))

        not_in_event_ser = meas_ser.transform(obs_out_event)
        event_id_ser = not_in_event_ser.cumsum() + 1
        event_id_ser.loc[not_in_event_ser] = 0
        event_df = pd.DataFrame(event_id_ser.rename("event_id")).reset_index()

        if max_time_elapsed == 0:
            return event_df["event_id"].values
        else:
            # Group events that are near to one another in time
            def increment_event_id_group(x):
                return (x > max_time_elapsed) | np.isnan(x)

            group_df = (
                event_df.query("event_id > 0")
                .groupby("event_id", as_index=False)
                .agg(
                    event_start_time=pd.NamedAgg("time", "first"),
                    event_end_time=pd.NamedAgg("time", "last"),
                )
                .assign(
                    seconds_elapsed=lambda x: (
                        x["event_start_time"] - x["event_end_time"].shift(1)
                    ).dt.total_seconds()
                )
                .assign(
                    event_id_group=lambda x: (
                        increment_event_id_group(x["seconds_elapsed"])
                    ).cumsum()
                )
            )
            group_df = group_df.drop(
                columns=np.setdiff1d(group_df.columns, ["event_id", "event_id_group"])
            )

            # Apply event groups to measurements
            event_df = event_df.merge(group_df, how="left", on="event_id")
            event_df["event_id_group_bfill"] = event_df["event_id_group"].bfill()
            event_df["event_id_group_ffill"] = event_df["event_id_group"].ffill()
            event_df["fill_match"] = (
                event_df["event_id_group_bfill"] == event_df["event_id_group_ffill"]
            )
            event_df.loc[event_df["fill_match"], "event_id_group"] = event_df.loc[
                event_df["fill_match"], "event_id_group_bfill"
            ]
            event_df["event_id_group"] = (
                event_df["event_id_group"].replace(np.nan, 0).astype(int)
            )
            return event_df["event_id_group"].values

    event_df["event_id_group"] = (
        event_df["obs"]
        .groupby("grp")
        .transform(
            get_events_geyser,
            obs_in_event=obs_in_event,
            max_time_elapsed=max_time_elapsed,
        )
    )

    # Create unique event ids across all geysers
    event_df = event_df.reset_index()
    uniq_df = event_df.loc[
        event_df["event_id_group"] > 0, ["grp", "event_id_group"]
    ].drop_duplicates()
    uniq_df["event_id_uniq"] = np.arange(uniq_df.shape[0]) + 1
    event_df = event_df.merge(uniq_df, how="left", on=["grp", "event_id_group"])

    # Clean up for return
    event_df["event_id_uniq"] = event_df["event_id_uniq"].replace(np.nan, 0).astype(int)

    return event_df["event_id_uniq"].values
