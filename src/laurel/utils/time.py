"""Time-zone conversion, local-time computation, and circular time statistics.

This module provides utilities for working with timestamped dwell and event
data across multiple U.S. time zones.  Because vehicle dwells are recorded in
UTC but load profiles must be expressed in local time, accurate time-zone
lookup and conversion are central to the pipeline.

Key functions:

- :func:`calc_time_zones_from_hexes` — look up the IANA time-zone string for
  each row based on its H3 hexagon, using ``tzfpy`` for point-in-polygon
  queries.
- :func:`calc_local_time` — group rows by time zone and convert UTC timestamps
  to timezone-naive local times.
- :func:`calc_avg_time_of_day` — compute the circular mean of a time-of-day
  array (Numba JIT), correctly handling the midnight wrap-around.

Key design decisions
--------------------
- **Unique-hex caching**: :func:`calc_time_zones_from_hexes` resolves time zones
  only for the unique set of hexagon IDs before merging back, avoiding redundant
  ``tzfpy`` calls for the many rows that share the same hex.
- **Timezone-naive output**: :func:`_get_local_time_by_tz` strips the timezone
  info after conversion (``dt.tz_localize(None)``), so downstream pandas
  operations can compare timestamps without mixed-tz errors.
- **Circular statistics**: :func:`calc_avg_time_of_day` uses the
  unit-circle projection (cos/sin → arctan2) to compute a mean that wraps
  correctly at midnight; the same projection yields a meaningful standard
  deviation via the angular residuals.
"""

from collections.abc import Callable

import h3.api.basic_str as h3_str
import h3.api.numpy_int as h3
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
from tzfpy import get_tz

SECS_PER_HOUR = 3600
HOURS_PER_WEEK = 168


def calc_time_zones_from_hexes(
    df: pd.DataFrame,
    hex_col: str,
    tz_col: str = "tz",
) -> pd.DataFrame:
    """Assign an IANA time-zone string to each row based on its H3 hexagon.

    Converts hex IDs to centroid coordinates, queries ``tzfpy`` for the
    corresponding time zone, then merges the result back onto ``df``.  To
    minimise expensive point-in-polygon queries, only the *unique* hex values
    are resolved; rows sharing the same hex reuse the cached result.

    Args:
        df: DataFrame containing a column of H3 integer cell IDs.
        hex_col: Name of the column holding H3 integer cell IDs.
        tz_col: Name of the output column to write the IANA timezone string into.
            Defaults to ``"tz"``.

    Returns:
        ``df`` with a new ``tz_col`` column of ``pd.Categorical`` timezone strings
        and the original index restored.
    """
    orig_idx = df.index.names
    if orig_idx != [None]:
        df = df.reset_index()
    # Getting unique hexes
    str_col = f"{hex_col}_str"
    df[str_col] = df[hex_col].transform(h3.int_to_str)
    hex_arr = df[str_col].unique()
    # Identifying time zones for unique hexes
    hexes = pd.DataFrame(data=hex_arr, columns=[str_col])
    hexes[tz_col] = hexes[str_col].transform(get_timezone_from_hex)
    hexes[tz_col] = pd.Categorical(hexes[tz_col])
    # Merging time zones back onto original dataframe
    hexes = hexes.set_index(str_col)
    df = df.merge(hexes, how="left", left_on=str_col, right_index=True)
    df = df.drop(columns=[str_col])
    if orig_idx != [None]:
        df = df.set_index(orig_idx)
    return df


def get_timezone_from_hex(hex: int | str) -> str:
    """Return the IANA timezone string for the centroid of an H3 hexagon.

    Args:
        hex: An H3 cell ID as either a ``numpy.uint64`` integer or a hex string.

    Returns:
        IANA timezone string (e.g. ``"America/Los_Angeles"``).

    Raises:
        RuntimeError: If ``hex`` is neither an integer nor a string.
    """
    if isinstance(hex, int):
        lat, lng = h3.cell_to_latlng(hex)
    elif isinstance(hex, str):
        lat, lng = h3_str.cell_to_latlng(hex)
    else:
        raise RuntimeError("Hex argument came in as neither a string nor an integer.")
    tz_str = get_tz(lng=lng, lat=lat)
    return tz_str


def get_timezones(hexes: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Kedro node wrapper: add a timezone column to a hexagon correspondence table.

    Args:
        hexes: DataFrame with at least one column of H3 cell IDs.
        params: Configuration dict with the following key:

            - **hex_col** (``str``): Name of the H3 cell ID column.

    Returns:
        ``hexes`` with a new ``"tz"`` column of IANA timezone strings.
    """
    hexes = calc_time_zones_from_hexes(df=hexes, hex_col=params["hex_col"])
    return hexes


def calc_local_time(
    df: pd.DataFrame,
    time_cols: str | list[str],
    local_cols: str | list[str],
    tz_col: str,
    sort_col: str = None,
    grp_cols: str | list[str] = None,
) -> pd.DataFrame:
    """Add timezone-naive local-time columns to a DataFrame of UTC timestamps.

    Groups rows by timezone (and optionally by additional group columns), then
    calls :func:`_get_local_time_by_tz` on each group to convert the UTC
    ``time_cols`` to their local equivalents.  Optionally re-sorts each group
    by ``sort_col`` after conversion.

    Args:
        df: DataFrame containing UTC timestamp columns and a timezone column.
        time_cols: UTC timestamp column name(s) to convert.
        local_cols: Output column name(s), in the same order as ``time_cols``.
        tz_col: Column containing IANA timezone strings (e.g. ``"tz"``).
        sort_col: If provided, rows within each group are sorted by this column
            after local-time conversion.
        grp_cols: Additional column(s) to group by before the timezone grouping.
            Useful when each vehicle/region should be treated independently.

    Returns:
        ``df`` with new columns given by ``local_cols`` containing timezone-naive
        local timestamps.

    Raises:
        RuntimeError: If ``grp_cols`` is not a string, list, or ``None``.
    """
    if grp_cols is None:
        grouper = [tz_col]
    elif isinstance(grp_cols, str):
        grouper = [grp_cols, tz_col]
    elif isinstance(grp_cols, list):
        grouper = grp_cols + [tz_col]
    else:
        raise RuntimeError("grp_cols argument must be a string or list.")

    if isinstance(time_cols, str):
        time_cols = [time_cols]
    if isinstance(local_cols, str):
        local_cols = [local_cols]
    for col in local_cols:
        # May need to be adjusted for different attribute dtypes
        df[col] = pd.Timestamp(0)
    # Building local time columns
    tqdm.pandas()
    df = df.groupby(
        grouper, group_keys=False, sort=False, observed=True
    ).progress_apply(
        lambda g: _get_local_time_by_tz(
            g,
            tz=g.name if isinstance(g.name, str) else g.name[-1],
            utc_cols=time_cols,
            local_cols=local_cols,
        )
    )

    if sort_col is not None:
        # Sorting within each group
        df = df.groupby(
            grp_cols, group_keys=False, sort=False, observed=True
        ).progress_apply(lambda grp: grp.sort_values(sort_col))
    return df


def _get_local_time_by_tz(
    grp: pd.DataFrame,
    tz: str,
    utc_cols: str | list[str],
    local_cols: str | list[str],
) -> pd.DataFrame:
    """Get a timezone-naive local time from a UTC datetime column and a timezone.

    This function assumes that there is a single time zone across the dataframe. If this
    is not the case, then group by time zone and then pass to this function.
    """
    if isinstance(utc_cols, str):
        utc_cols = [utc_cols]
    if isinstance(local_cols, str):
        local_cols = [local_cols]
    if len(utc_cols) != len(local_cols):
        raise RuntimeError(
            "utc_cols and local_cols arguments must have the same length."
        )

    for tcol, lcol in zip(utc_cols, local_cols):
        grp.loc[:, lcol] = grp[tcol].dt.tz_convert(tz).dt.tz_localize(None)
    return grp


def calc_time_attrs(df: pd.DataFrame, time_col: str, attrs: list[str]) -> pd.DataFrame:
    """Add datetime accessor attributes as new columns (e.g. ``hour``, ``dayofweek``).

    For each attribute name in ``attrs``, accesses ``df[time_col].dt.<attr>``
    and writes the result to a new column named ``{time_col}_{attr}``.

    Args:
        df: DataFrame with a datetime column.
        time_col: Name of the datetime column to extract attributes from.
        attrs: List of ``pandas.DatetimeIndex`` accessor attribute names
            (e.g. ``["hour", "dayofweek", "month"]``).

    Returns:
        ``df`` with one new column per entry in ``attrs``.
    """
    if isinstance(attrs, str):
        attrs = [attrs]

    for a in attrs:
        new_name = f"{time_col}_{a}"
        df.loc[:, new_name] = getattr(df[time_col].dt, a)
    return df


def total_time_units(s: pd.Series, unit: str) -> pd.Series:
    """Convert a Series of timedeltas to fractional time units.

    Args:
        s: Series of ``pd.Timedelta`` values.
        unit: Pandas offset string defining the unit (e.g. ``"1h"``, ``"1min"``).

    Returns:
        Float Series of total elapsed units.
    """
    return s.dt.total_seconds() / pd.Timedelta(value=unit).total_seconds()


def total_hours(s: pd.Series) -> pd.Series:
    """Convert a Series of timedeltas to fractional hours."""
    return total_time_units(s, unit="1h")


def get_total_time_units_filtered(
    start: pd.Timestamp,
    end: pd.Timestamp,
    unit: str,
    filterer: Callable[["pd.Series[pd.Timestamp]"], "pd.Series[bool]"] | None = None,
) -> int:
    """Count the number of time-unit boundaries between ``start`` and ``end``, optionally filtered.

    Generates a date range from ``floor(start)`` to ``ceil(end)`` at ``unit``
    frequency and counts the matching timestamps.  The optional ``filterer``
    callable allows arbitrary masks (e.g. keep only weekdays) to be applied
    before counting.

    Args:
        start: Observation start timestamp.
        end: Observation end timestamp.
        unit: Pandas offset string for the time unit (e.g. ``"1h"``, ``"1d"``).
        filterer: Optional callable that accepts a ``Series[Timestamp]`` and
            returns a boolean ``Series``.  Only ``True`` timestamps are counted.

    Returns:
        Integer count of qualifying time-unit boundaries in the window.
    """
    end_ceil = end.ceil(unit)
    start_floor = start.floor(unit)
    times = pd.date_range(start=start_floor, end=end_ceil, freq=unit).to_series()

    if filterer is not None:
        tot_t_units = filterer(times).sum()
    else:
        tot_t_units = len(times)
    return tot_t_units


@jit
def calc_avg_time_of_day(t: np.ndarray, full_day: float) -> float:
    """Calculate the average time of day, dealing with midnight.

    Note: When two times are diametrically opposed to each other, the default average
    is the greater of the two possible averages.

    Args:
        t: the array of times-of-day
        full_day: the maximum value that t takes before resetting to zero (e.g. 24 for
            24 hours in a day)

    Returns: float of average hour-of-day in t
    """
    ratio = 2 * np.pi / full_day
    rads = ratio * t
    xs, ys = np.cos(rads), np.sin(rads)
    xavg, yavg = np.mean(xs), np.mean(ys)
    ravg = np.arctan2(yavg, xavg)
    tavg = ravg / ratio
    if tavg < 0:
        tavg += full_day

    devs = np.mod(rads - ravg, 2 * np.pi)
    devs = np.where(devs > np.pi, devs - 2 * np.pi, devs)
    devs = np.where(devs < -np.pi, devs + 2 * np.pi, devs)
    rstd = np.std(devs)
    tstd = rstd / ratio
    return tavg, tstd
