# # Calculate Energy Consumption Distribution from RoL-E
#
# *by Fletcher Passow*
#
# July 2023

import geopandas as gpd
import numpy as np
import pandas as pd

from .events import get_events

METERS_PER_MILE = 1609.344
SECONDS_PER_HOUR = 3600


def calc_energy_consump_rate(
    energy_df: pd.DataFrame, gps_df: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Calculate energy consumption in kWh per mile."""
    # ## Calculate driving sessions from energy usage
    energy_df["active_energy"] = (
        energy_df["driving_energy"]
        - energy_df["energy_regen"]
        + energy_df["idle_energy"]
    )
    energy_df["drive_event_id"] = get_events(
        obs_ser=energy_df["active_energy"],
        time_ser=energy_df.index.get_level_values("timestamp"),
        grp_ser=energy_df.index.get_level_values("truck_id"),
        obs_in_event=lambda x: x > 0,
        max_time_elapsed=params["allow_sess_idle_time_hrs"] * SECONDS_PER_HOUR,
    )

    # ### Drop driving sessions which are too long or too short
    drive_event_df = energy_df.groupby(["truck_id", "drive_event_id"]).agg(
        duration_hrs=pd.NamedAgg("time_diff_sec", lambda x: x.sum() / SECONDS_PER_HOUR),
        total_energy_kwh=pd.NamedAgg("active_energy", "sum"),
    )
    too_short_ser = drive_event_df["duration_hrs"] < params["min_drive_sess_length_hrs"]
    too_long_ser = drive_event_df["duration_hrs"] > params["max_drive_sess_length_hrs"]
    is_zero_ser = drive_event_df.index.get_level_values("drive_event_id") == 0
    joint_filter_ser = np.logical_or(
        np.logical_or(too_short_ser, too_long_ser), is_zero_ser
    )
    drop_idx = drive_event_df.loc[joint_filter_ser].index
    select_drive_event_df = drive_event_df.drop(index=drop_idx)
    select_drive_event_df["new_drive_event_id"] = np.arange(
        1, select_drive_event_df.shape[0] + 1
    )
    join_df = select_drive_event_df.loc[:, ["new_drive_event_id"]].reset_index()
    energy_df = energy_df.reset_index().merge(
        join_df,
        how="left",
        left_on=["truck_id", "drive_event_id"],
        right_on=["truck_id", "drive_event_id"],
    )
    energy_df["new_drive_event_id"] = (
        energy_df["new_drive_event_id"].fillna(0).astype(int)
    )
    energy_df = energy_df.drop(columns=["drive_event_id"])
    energy_df = energy_df.rename(columns={"new_drive_event_id": "drive_event_id"})

    # Get distance traveled by time step
    gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"], utc=True, format="mixed")
    gps_df = gps_df.drop_duplicates(subset=["truck_id", "timestamp"], keep="last")
    gps_df = gps_df.sort_values(["truck_id", "timestamp"])
    gps_df["dist_traveled_mi"] = calc_distances_between_obs(
        lat=gps_df["latitude"],
        lon=gps_df["longitude"],
        grp=gps_df["truck_id"],
        crs=params["coord_ref_sys"],
    )
    gps_df = gps_df.loc[:, ["truck_id", "timestamp", "dist_traveled_mi"]]

    # Combine energy and GPS and calculate consumption
    gps_df = gps_df.set_index(["truck_id", "timestamp"])
    gps_df["drive_event_id"] = get_aligned_event_ids(comp=gps_df, ref=energy_df)
    idx_cols = ["truck_id", "drive_event_id"]
    energy_merge_df = energy_df.groupby(idx_cols).agg(
        total_energy_kwh=pd.NamedAgg("active_energy", "sum")
    )
    dist_merge_df = gps_df.groupby(idx_cols).agg(
        total_dist_mi=pd.NamedAgg("dist_traveled_mi", "sum")
    )
    # Using inner join here because drive events without any distance are
    # probably due to energy readings extending farther than GPS readings
    consump_df = energy_merge_df.merge(dist_merge_df, on=idx_cols, how="inner")
    consump_df = consump_df.query("drive_event_id > 0")
    drop_idx = consump_df.loc[
        consump_df["total_dist_mi"] < params["min_drive_sess_dist_miles"]
    ].index
    consump_df = consump_df.drop(index=drop_idx)
    consump_df["consump_kwh_per_mi"] = (
        consump_df["total_energy_kwh"] / consump_df["total_dist_mi"]
    )
    return consump_df


def get_aligned_event_ids(comp: pd.DataFrame, ref: pd.DataFrame) -> pd.Series:
    """Apply event IDs from a reference grouped time series to an unaligned
    comparison grouped time series by matching the comparison time series
    observations to those in the which are closest beginning and ending times of
    events from the reference.

    Args:
        comp:
            comparison dataframe to give event IDs for, must have columns
            ["truck_id", "timestamp"]
        ref:
            reference dataframe to take event IDs from, must have columns
            ["truck_id", "timestamp", "drive_event_id"]

    Returns: Series of event IDs of the same length as the comp dataframe
    """
    # Create dataframe of drive event beginning and ending times
    ref["drive_event_id_shifted"] = ref["drive_event_id"].shift(-1)
    begin_ser = ref.groupby(["truck_id", "drive_event_id_shifted"])["timestamp"].first()
    end_ser = ref.groupby(["truck_id", "drive_event_id"])["timestamp"].last()
    event_times = pd.DataFrame({"begin_time": begin_ser, "end_time": end_ser})
    event_times.index = event_times.index.rename(["truck_id", "drive_event_id"])
    is_zero_ser = event_times.index.get_level_values("drive_event_id") == 0
    drop_idx = event_times.loc[is_zero_ser].index
    event_times = event_times.drop(index=drop_idx)
    event_times = event_times.reset_index()
    event_times["drive_event_id"] = event_times["drive_event_id"].astype(int)

    # Create dataframe of comp observations to join on
    comp_times = comp.reset_index().loc[:, ["truck_id", "timestamp"]]
    comp_times = comp_times.reset_index()
    comp_times = comp_times.rename(columns={"index": "obs_id"})
    comp_times = comp_times.sort_values("timestamp")

    # Identify first and last comp observations within each event
    begin_joined_df = pd.merge_asof(
        event_times.sort_values("begin_time"),
        comp_times,
        by="truck_id",
        left_on="begin_time",
        right_on="timestamp",
        direction="forward",
    )
    begin_joined_df = begin_joined_df.drop_duplicates(
        ["truck_id", "timestamp"], keep="last"
    )
    end_joined_df = pd.merge_asof(
        event_times.sort_values("end_time"),
        comp_times,
        by="truck_id",
        left_on="end_time",
        right_on="timestamp",
        direction="backward",
    )
    end_joined_df = end_joined_df.drop_duplicates(
        ["truck_id", "timestamp"], keep="first"
    )

    # Join event IDs onto comp observation extreme times
    joiner_cols = ["truck_id", "timestamp", "drive_event_id"]
    comp_events = comp_times.merge(
        begin_joined_df.loc[:, joiner_cols], on=["truck_id", "timestamp"], how="left"
    )
    comp_events = comp_events.merge(
        end_joined_df.loc[:, joiner_cols],
        on=["truck_id", "timestamp"],
        how="left",
        suffixes=["_begin", "_end"],
    )

    # Apply event IDs to in-between observation times
    comp_events = comp_events.set_index(["truck_id", "timestamp"]).sort_index()
    begin_fill = comp_events["drive_event_id_begin"].ffill().fillna(0)
    end_fill = comp_events["drive_event_id_end"].bfill().fillna(0)
    matching = begin_fill == end_fill
    comp_events["drive_event_id"] = (begin_fill * matching).astype(int)
    return comp_events["drive_event_id"]


def calc_utm_distance(ser_a: gpd.GeoSeries, ser_b: gpd.GeoSeries) -> pd.Series:
    """Calculate distance in meters between two GeoSeries using the best-fit UTM CRS.

    Calculated as ser_a - ser_b.

    Args:
        ser_a:
            series of geographic objects forming the comparison group
        ser_b:
            series of geographic objects forming the reference group

    Returns:
        series of distances in units of meters, since UTM projections are in
        units of meters.
    """
    unified_ser = gpd.GeoSeries(pd.concat([ser_a, ser_b], axis=0))
    crs = unified_ser.estimate_utm_crs()
    ser_a_proj = ser_a.to_crs(crs)
    ser_b_proj = ser_b.to_crs(crs)
    dist_ser = ser_a_proj.distance(ser_b_proj)
    return dist_ser


def calc_distances_between_obs(
    lat: pd.Series, lon: pd.Series, grp: pd.Series, crs: int
) -> pd.Series:
    """Calculate distances in miles between successive observations of latitude
    and longitude.

    Assumes that the dataset is already sorted by time.

    Args:
        lat:
            latitude coordinate for each observation
        lon:
            longitude coordinate for each observation
        grp:
            grouping for each observation, to prevent calculation of distance
            between different groups, where a group might be a vehicle-day
        time:
            series of timestamps for each observation, used for sorting
        crs:
            the coordinate reference system to use for the lat, lon pairs

    Returns: distances between each pair of observations
    """
    gps = gpd.GeoDataFrame(
        {
            "grp": grp,
        },
        geometry=gpd.points_from_xy(lon, lat),
        crs=crs,
    )

    def calc_dist_to_next(grp: gpd.GeoSeries):
        return calc_utm_distance(grp.geometry, grp.geometry.shift(1))

    gps["dist"] = gps.groupby("grp", group_keys=False).apply(calc_dist_to_next)
    gps["dist"] = gps["dist"].fillna(0)
    gps["dist_traveled_mi"] = gps["dist"] / METERS_PER_MILE
    return gps["dist_traveled_mi"].values
