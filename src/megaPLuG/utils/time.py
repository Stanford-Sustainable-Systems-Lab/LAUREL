import logging

import h3.api.basic_str as h3_str
import h3.api.numpy_int as h3
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
from tzfpy import get_tz

logger = logging.getLogger(__name__)

SECS_PER_HOUR = 3600


def calc_time_zones_from_hexes(
    df: pd.DataFrame,
    hex_col: str,
    tz_col: str = "tz",
) -> pd.DataFrame:
    """Find the time zone for each row of a dataframe based on an H3 hexagon column."""
    orig_idx = df.index.names
    if orig_idx != [None]:
        df = df.reset_index()
    logger.info("Getting unique hexes")
    str_col = f"{hex_col}_str"
    df[str_col] = df[hex_col].transform(h3.int_to_str)
    hex_arr = df[str_col].unique()
    logger.info("Identifying time zones for unique hexes")
    hexes = pd.DataFrame(data=hex_arr, columns=[str_col])
    hexes[tz_col] = hexes[str_col].transform(get_timezone_from_hex)
    logger.info("Merging time zones back onto original dataframe")
    hexes = hexes.set_index(str_col)
    df = df.merge(hexes, how="left", left_on=str_col, right_index=True)
    df = df.drop(columns=[str_col])
    if orig_idx != [None]:
        df = df.set_index(orig_idx)
    return df


def get_timezone_from_hex(hex: int | str) -> str:
    """Get the timezone string fror the h3 hexagon."""
    if isinstance(hex, int):
        lat, lng = h3.cell_to_latlng(hex)
    elif isinstance(hex, str):
        lat, lng = h3_str.cell_to_latlng(hex)
    else:
        raise RuntimeError("Hex argument came in as neither a string nor an integer.")
    tz_str = get_tz(lng=lng, lat=lat)
    return tz_str


def get_timezones(hexes: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Get the timezones based on the hexagons."""
    hexes = calc_time_zones_from_hexes(df=hexes, hex_col=params["hex_col"])
    return hexes


def calc_local_time_attrs(
    df: pd.DataFrame,
    time_cols: str | list[str],
    attrs: str | list[str],
    tz_col: str,
    sort_col: str = None,
    grp_cols: str | list[str] = None,
) -> pd.DataFrame:
    """Modifies the passed dataframe to also include local time attribute columns.

    See the pandas.Timestamp documentation for possible attributes.
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
    for tcol in time_cols:
        for a in attrs:
            new_name = get_local_time_attr_col_name(time_col=tcol, attr_name=a)
            df[new_name] = 0  # May need to be adjusted for different attribute dtypes
    logger.info("Building time attribute columns")
    tqdm.pandas()
    df = df.groupby(grouper, group_keys=False, sort=False).progress_apply(
        lambda g: _get_local_time_attr_by_tz(
            g,
            tz=g.name if isinstance(g.name, str) else g.name[-1],
            utc_cols=time_cols,
            attrs=attrs,
        )
    )

    if sort_col is not None:
        logger.info("Sorting within each group.")
        df = df.groupby(grp_cols, group_keys=False, sort=False).progress_apply(
            lambda grp: grp.sort_values(sort_col)
        )
    return df


def _get_local_time_attr_by_tz(
    grp: pd.DataFrame,
    tz: str,
    utc_cols: str | list[str],
    attrs: str | list[str],
) -> pd.DataFrame:
    """Get a time-based metric, like day, day_of_year, etc. from a UTC datetime column
    and a timezone.

    This function assumes that there is a single time zone across the dataframe. If this
    is not the case, then group by time zone and then pass to this function.
    """
    if isinstance(utc_cols, str):
        utc_cols = [utc_cols]
    if isinstance(attrs, str):
        attrs = [attrs]

    for tcol in utc_cols:
        local_time_ser = grp[tcol].dt.tz_convert(tz)
        for a in attrs:
            new_name = get_local_time_attr_col_name(time_col=tcol, attr_name=a)
            grp.loc[:, new_name] = getattr(local_time_ser.dt, a)
    return grp


def get_local_time_attr_col_name(time_col: str, attr_name: str) -> str:
    return f"{time_col}_local_{attr_name}"


def total_hours(s: pd.Series) -> pd.Series:
    """Get the total number of hours from a series of timedeltas."""
    return s.dt.total_seconds() / SECS_PER_HOUR


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
