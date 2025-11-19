"""
This is a boilerplate pipeline 'describe_vehicles'
generated using Kedro 0.19.3
"""

import logging

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
from dask.diagnostics.progress import ProgressBar

from megaplug.models.dwell_sets import DwellSet
from megaplug.utils.geo import (
    METERS_PER_MILE,
    find_time_weighted_centers,
)
from megaplug.utils.h3 import add_geometries
from megaplug.utils.params import build_df_from_dict
from megaplug.utils.time import total_hours

logger = logging.getLogger(__name__)


def strip_vehicle_attrs(
    trips: dd.DataFrame, params: dict
) -> tuple[dd.DataFrame, pd.DataFrame]:
    """Get vehicle-specific attributes which stay constant."""
    n_trips_by_veh = trips[params["veh_id_col"]].value_counts().compute()
    drop_idx = n_trips_by_veh.loc[n_trips_by_veh < params["min_trips_per_veh"]].index

    veh_cols = [params["veh_id_col"]] + params["veh_attr_cols"]
    vehs = trips.loc[:, veh_cols].drop_duplicates().compute()
    vehs = vehs.set_index(params["veh_id_col"]).sort_index()
    vehs = vehs.drop(index=drop_idx)
    return vehs


def partition_dwellset(dw: DwellSet, params: dict) -> DwellSet:
    """Re-partition the trips to save to disk, in preparation for routing."""
    if dw.is_dask:
        dw.data = dw.data.repartition(npartitions=params["n_partitions"])
    return dw


def get_vehicle_observation_frames(
    vehs: pd.DataFrame, dw: DwellSet, params: dict
) -> pd.DataFrame:
    """Get the total time and mileage over which each vehicle is observed."""
    if not dw.is_dask:
        dw.sort_by_veh_time()
    else:
        logger.warning("Assuming that the Dask-based DwellSet is sorted.")
    veh_obs = dw.data.groupby(dw.veh).agg(
        obs_time_first=pd.NamedAgg(dw.start, "first"),
        obs_hex_first=pd.NamedAgg(dw.hex, "first"),
        obs_time_last=pd.NamedAgg(dw.end, "last"),
        obs_hex_last=pd.NamedAgg(dw.hex, "last"),
        dist_traveled_col=pd.NamedAgg(dw.trip_dist, "sum"),
    )
    veh_obs["obs_time_col"] = veh_obs["obs_time_last"] - veh_obs["obs_time_first"]
    veh_obs = veh_obs.rename(columns=params["column_namer"])
    if dw.is_dask:
        with ProgressBar():
            veh_obs = veh_obs.compute()
    vehs = vehs.merge(veh_obs, how="left", on=dw.veh)
    return vehs


def classify_vehicles(
    vehs: pd.DataFrame, veh_locs: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Classify vehicles by their route type, home base status, etc."""
    veh_loc_cts = veh_locs.groupby(params["veh_col"], sort=False)[
        params["loc_col"]
    ].value_counts()
    veh_loc_cts = veh_loc_cts.unstack(params["loc_col"])
    veh_loc_cts["has_home_base"] = veh_loc_cts[params["base_location_type"]] > 0
    vehs = vehs.merge(
        veh_loc_cts.loc[:, ["has_home_base"]], how="left", on=params["veh_col"]
    )
    vehs.loc[vehs["has_home_base"].isna(), "has_home_base"] = False
    vehs["has_home_base"] = vehs["has_home_base"].astype(bool)
    return vehs


def mark_weight_class_group(vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Mark the vehicles with VIUS weight class groups."""
    wgt_corresp = build_df_from_dict(
        params["values"],
        id_cols=params["id_columns"],
        value_col=params["value_col"],
    )
    orig_idx = vehs.index.names
    vehs = vehs.reset_index()
    vehs = vehs.merge(wgt_corresp, how="left", on=params["id_columns"])
    vehs = vehs.set_index(orig_idx)
    return vehs


def mark_vehicle_centers(
    vehs: pd.DataFrame, veh_locs: pd.DataFrame, params: dict, dwell_params: dict
) -> pd.DataFrame:
    """Mark the characteristic center coordinates for each vehicle.

    If the vehicle has any depot locations, then select one of those based on priority
    parameters and sorting.
    """
    veh_col = dwell_params["veh"]
    hex_col = dwell_params["hex"]
    loc_col = params["location_col"]

    vlocs = veh_locs.reset_index()
    par_sort = params["sort_primary_locations_to_top"]
    vlocs = vlocs.sort_values(by=par_sort["columns"], ascending=par_sort["ascending"])
    is_base_loc_type = vlocs[loc_col] == params["base_location_type"]
    bases = vlocs.loc[is_base_loc_type, [veh_col, hex_col]]
    bases = bases.drop_duplicates(subset=[veh_col], keep="first")
    bases[hex_col] = bases[hex_col].astype(str)
    bases = bases.rename(columns={hex_col: params["home_base_col"]})

    vehs = vehs.merge(bases, how="left", on=veh_col)
    vehs = vehs.set_index(veh_col)
    vehs[params["home_base_col"]] = vehs[params["home_base_col"]].fillna(
        str(params["nan_int"])
    )  # Using zero as a NaN to preserve ints
    vehs[params["home_base_col"]] = vehs[params["home_base_col"]].astype(int)
    return vehs


def filter_dwells_for_op_segment(dw: DwellSet) -> DwellSet:
    """Filter down the dwells in preparation for computing the operating segment."""
    # Filter out all optional stops, which have the same start and end time (zero duration)
    dw.data = dw.data.loc[dw.data[dw.end] != dw.data[dw.start]]
    return dw


def spatialize_dwells(dw: DwellSet) -> DwellSet:
    """Add geometries to dwells."""
    if not isinstance(dw.data, gpd.GeoDataFrame):
        logger.info("Converting DwellSet data to GeoDataFrame.")
        dw.to_geodataframe()

    return dw


def get_operating_segment(
    vehs: pd.DataFrame, dw: DwellSet, params: dict
) -> pd.DataFrame:
    """Calculate the operating segment of each vehicle based primary operating distance.
    If a vehicle does not have a home base, then use the time-weighted center as the
    reference point instead.
    """
    if not isinstance(dw.data, gpd.GeoDataFrame):
        logger.info("Converting DwellSet data to GeoDataFrame.")
        dw.to_geodataframe()

    dw.data = dw.data.to_crs(params["proj_crs"])
    logger.info(
        "Get the time-weighted center location for vehicles without identifiable depots."
    )
    veh_no_home = vehs[params["home_base_col"]] == params["nan_int"]
    vehs_wo_homes = vehs.loc[veh_no_home].index
    dw_wo_homes = dw.copy_without_data()
    dw_wo_homes.data = dw.data.loc[vehs_wo_homes]

    dw_wo_homes.data["dwell_hrs"] = total_hours(
        dw_wo_homes.data[dw.end] - dw_wo_homes.data[dw.start]
    )
    centers = find_time_weighted_centers(
        gdf=dw_wo_homes.data,
        grp_col=dw_wo_homes.veh,
        weight_col="dwell_hrs",
    )

    bases = vehs.loc[~veh_no_home, :]
    bases = add_geometries(data=bases, hex_col=params["home_base_col"])
    bases = bases.to_crs(params["proj_crs"])
    all_centers = pd.concat([centers.geometry, bases.geometry], axis=0)
    all_centers.name = "center"

    logger.info("Calculating distances from center location.")
    dw.data = dw.data.merge(all_centers, how="left", on=dw.veh)
    dw.data["rad_miles"] = (
        dw.data.geometry.distance(dw.data["center"]) / METERS_PER_MILE
    )
    max_rad = dw.data["rad_miles"].max()
    bins = params["radius_bin_low_bounds_miles"]
    dw.data["rad_miles_bin"] = pd.cut(
        dw.data["rad_miles"],
        bins=list(bins.values()) + [max_rad],
        labels=list(bins.keys()),
        include_lowest=True,
    )

    segs = dw.data.groupby([dw.veh, "rad_miles_bin"])[dw.trip_dist].sum()
    segs = segs.unstack(level="rad_miles_bin", fill_value=0.0)
    segs[params["segment_col"]] = segs.idxmax(axis=1)

    vehs = vehs.merge(segs.loc[:, params["segment_col"]], how="left", on=dw.veh)
    return vehs


def mark_location_regions(
    vehs: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    params: dict,
) -> pd.DataFrame:
    """Mark the region of chosen locations by intersecting location points with region polygons."""
    veh_col = params["veh_col"]
    loc_reg_col = params["location_region_col"]

    bases = vehs.loc[vehs[params["home_base_col"]] != params["nan_int"], :]
    bases = add_geometries(bases, hex_col=params["home_base_col"])
    bases = bases.to_crs(regions.crs)
    mrg = regions.loc[:, [params["region_name_col"], regions.geometry.name]]
    base_regs = bases.sjoin(mrg, how="left")
    base_regs = base_regs.rename(columns={params["region_name_col"]: loc_reg_col})
    base_regs = base_regs.loc[:, [loc_reg_col]]
    vehs = vehs.merge(base_regs, how="left", on=veh_col)
    vehs[loc_reg_col] = vehs[loc_reg_col].fillna(params["na_region_fill"])
    return vehs
