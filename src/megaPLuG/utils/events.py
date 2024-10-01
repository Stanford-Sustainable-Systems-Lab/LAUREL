"""
Utility functions for events
by Fletcher Passow
October 2024
"""

import logging

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_events(
    df: pd.DataFrame,
    include_col: str,
    dur_col: str,
    grp_col: str,
    out_col: str = "event_id",
    max_time_elapsed: pd.Timedelta = pd.Timedelta(0, "s"),
) -> pd.DataFrame:
    """Get an event id column which gives the indices for events based on periodic
    observations.

    Events are defined as semi-contiguous stretches of a value matching the
    criterion given by obs_in_event. The breaks in an event can only be as long as
    max_time_elapsed.

    This function depends on sorting by time within each group, and that groups are
    each one block of rows.

    Args:
      df: DataFrame to add the event_id column to
      include_col: the name of a boolean series which is True for every row (observation) that
        should be in an event and False for every row (observation) that should NOT be
        in an event.
      dur_col: the name of a series of durations (as pandas TimeDeltas) of the events given in include_col
      grp_col: the name of a series of markers for groups of observations, usually this would be a series
        of integer ids, but if None, then all observations are assumed to
        come from the same group.
      max_time_elapsed: pandas TimeDelta giving the maximum time between events for them
              to be combined into a single event.

    Returns: a dataframe with a new column which gives an index of which event a particular
    observation is a part of (or zero if the observation is part of no event).
    Zero is used instead of a null value to mark the non-events because it allows
    the new column to be cast to an integer type, which facilitates merging.
    """
    df["secs_elapsed"] = df[dur_col].dt.total_seconds()
    df[out_col] = 0
    tqdm.pandas()
    df = df.groupby(grp_col, group_keys=False, sort=False).progress_apply(
        func=get_events_wrapper,
        include_col=include_col,
        secs_elapsed_col="secs_elapsed",
        out_col=out_col,
        max_time_elapsed=max_time_elapsed,
    )
    df = df.drop(columns=["secs_elapsed"])
    return df


def get_events_wrapper(
    grp: pd.DataFrame,
    include_col: str,
    secs_elapsed_col: str,
    out_col: str,
    max_time_elapsed: pd.Timedelta,
) -> pd.DataFrame:
    """Take the group and convert it into numba-compatible types."""
    grp[out_col] = get_events_core(
        include=grp[include_col].values,
        secs_elapsed=grp[secs_elapsed_col].values,
        max_secs_elapsed=max_time_elapsed.total_seconds(),
    )
    return grp


@njit
def get_events_core(
    include: np.ndarray[bool],
    secs_elapsed: np.ndarray[float],
    max_secs_elapsed: float = 0.0,
) -> np.ndarray:
    """Get a ndarray which gives the indices for events based on periodic
    observations.

    Events are defined as semi-contiguous stretches of a value matching the
    criterion given by obs_in_event. The breaks in an event can only be as long as
    max_time_elapsed.

    Assues interval-beginning time stamps, so events are marked at their beginning,
    with the duration assumed to follow.
    """
    nsteps = include.shape[0]
    cur_event = 0
    prev_incl = False
    secs_since_event_end = max_secs_elapsed + 1
    non_event_start_idx = 0
    event_ids = np.empty(nsteps, dtype=np.int64)
    for i in range(nsteps):
        incl = include[i]
        if incl:  # If this index is known to be in an event
            if not prev_incl:  # Coming into an event from a non-event
                if (
                    secs_since_event_end < max_secs_elapsed
                ):  # If not enough time has elapsed to declare a new event
                    event_ids[non_event_start_idx:i] = cur_event
                else:  # If enough time has elapsed to create a new event
                    event_ids[non_event_start_idx:i] = 0
                    cur_event += 1
            event_ids[i] = cur_event
        elif prev_incl:  # Just coming out of event
            non_event_start_idx = i
            secs_since_event_end = 0
        secs_since_event_end += secs_elapsed[
            i
        ]  # Assuming interval-beginning timestamps
        prev_incl = incl
    if not incl:  # If the final observation is not in a group
        event_ids[non_event_start_idx:nsteps] = 0  # Fill all tail event ids with zero
    return event_ids
