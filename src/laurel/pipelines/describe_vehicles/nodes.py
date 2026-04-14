"""Kedro pipeline nodes for the ``describe_vehicles`` pipeline (vehicle characterisation).

Characterises each vehicle in the telematics dataset along three dimensions
used by the electrification model: (1) the primary operating-distance class
(0–99, 100–249, 250–499, or 500+ miles), which determines the battery-range
option assigned during electrification; (2) the time-weighted geographic
center of operations, used as a spatial reference point; and (3) the
operating radius, which measures how far vehicles travel from their center.
These attributes feed directly into the ``electrify_trips`` pipeline.

Pipeline overview
-----------------
1. **filter_dwells_for_op_segment** — Removes zero-duration dwells
   (optional truck-stop records with identical start and end times) before
   computing distance-based vehicle statistics.
2. **spatialize_dwells** — Converts the ``DwellSet`` to a GeoDataFrame by
   adding H3-centroid geometries if they are not already present.
3. **partition_dwellset** — Re-partitions the Dask-backed ``DwellSet`` to
   the desired number of partitions before on-disk checkpointing.
4. **strip_vehicle_attrs** — Extracts vehicle-level constant attributes
   (weight class, cab type, etc.) and drops vehicles that appear fewer than
   ``min_trips_per_veh`` times.
5. **mark_weight_class_group** — Merges a VIUS-to-model weight-class
   correspondence onto the vehicle table.
6. **get_vehicle_observation_frames** — Records the first and last observed
   timestamps, total distance traveled, and first/last hexagon for each
   vehicle.
7. **get_operating_radius** — Computes the operating radius of each vehicle
   as the maximum haversine distance from the H3-centroid of any dwell to
   the vehicle's center.
8. **get_time_weighted_centers** — Computes the time-weighted geographic
   center of each vehicle's dwell history by projecting to a planar CRS and
   computing a dwell-duration-weighted centroid.
9. **get_operating_segment** — Assigns a primary operating-distance class
   (distance bin label) by binning dwell-to-center distances and selecting
   the bin that accounts for the most cumulative trip miles.

Key design decisions
--------------------
- **Dask/pandas dispatch**: Each spatial aggregation function contains an
  explicit ``dw.is_dask`` branch.  Dask partitions are mapped with
  ``map_partitions``; pandas is computed directly.  This lets the pipeline
  run in memory for debugging while retaining distributed-compute capability
  for the full 69,000-vehicle dataset.
- **Zero-duration dwell exclusion**: Optional truck-stop dwells inserted by
  ``compute_routes`` have zero duration and would contribute zero weight to
  a time-weighted centroid, but they introduce artificial distance-bin
  assignments.  Filtering them out before characterisation ensures that
  centroid and segment calculations reflect only genuine activity.
- **Deprecated functions**: ``classify_vehicles``, ``mark_vehicle_centers``,
  and ``mark_location_regions`` are preserved for potential future use but
  are not connected to the active pipeline as of 2025-11-19.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.
"""

from __future__ import annotations

import logging

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.diagnostics.progress import ProgressBar

from laurel.models.dwell_sets import DwellSet
from laurel.utils.geo import (
    METERS_PER_MILE,
    calc_operating_radius,
    find_time_weighted_centers,
)
from laurel.utils.h3 import H3_CRS, add_geometries
from laurel.utils.params import build_df_from_dict
from laurel.utils.time import total_hours

logger = logging.getLogger(__name__)


def filter_dwells_for_op_segment(dw: DwellSet) -> DwellSet:
    """Remove zero-duration dwells before computing operating-segment statistics.

    Optional truck-stop dwells inserted by the ``compute_routes`` pipeline
    have identical start and end timestamps (zero duration).  Leaving them in
    would artificially inflate visit counts along the route and introduce
    false intermediate locations into the distance-bin computation.

    Args:
        dw: Input ``DwellSet`` potentially containing optional stops.

    Returns:
        The ``DwellSet`` with all zero-duration rows dropped from ``dw.data``.
    """
    # Filter out all optional stops, which have the same start and end time (zero duration)
    dw.data = dw.data.loc[dw.data[dw.end] != dw.data[dw.start]]
    return dw


def spatialize_dwells(dw: DwellSet) -> DwellSet:
    """Convert the ``DwellSet`` data to a GeoDataFrame if not already spatial.

    Args:
        dw: Input ``DwellSet`` whose data may be a plain ``DataFrame`` or
            ``GeoDataFrame``.

    Returns:
        The ``DwellSet`` with ``dw.data`` guaranteed to be a ``GeoDataFrame``
        containing point geometries at each dwell's H3-hexagon centroid.
    """
    if not isinstance(dw.data, gpd.GeoDataFrame):
        logger.info("Converting DwellSet data to GeoDataFrame.")
        dw.to_geodataframe()

    return dw


def partition_dwellset(dw: DwellSet, params: dict) -> DwellSet:
    """Re-partition a Dask-backed ``DwellSet`` before writing to disk.

    Args:
        dw: Input ``DwellSet``.
        params: Pipeline parameters dict with keys:

            - ``n_partitions`` (int): target number of Dask partitions.

    Returns:
        The ``DwellSet`` with ``dw.data`` repartitioned (no-op for
        pandas-backed ``DwellSet`` instances).
    """
    if dw.is_dask:
        dw.data = dw.data.repartition(npartitions=params["n_partitions"])
    return dw


def strip_vehicle_attrs(
    trips: dd.DataFrame, params: dict
) -> tuple[dd.DataFrame, pd.DataFrame]:
    """Extract vehicle-level constant attributes and drop low-observation vehicles.

    Vehicles with fewer than ``min_trips_per_veh`` observed trips are excluded
    because they lack sufficient data to reliably estimate an operating-distance
    class or time-weighted center.

    Args:
        trips: Dask DataFrame of trip records with one row per trip.
        params: Pipeline parameters dict with keys:

            - ``veh_id_col`` (str): vehicle identifier column.
            - ``veh_attr_cols`` (list[str]): columns containing vehicle-level
              constant attributes (e.g., weight class, cab type).
            - ``min_trips_per_veh`` (int): minimum number of trips required to
              retain a vehicle.

    Returns:
        A ``pd.DataFrame`` indexed by ``veh_id_col``, one row per vehicle,
        with columns from ``veh_attr_cols``.
    """
    n_trips_by_veh = trips[params["veh_id_col"]].value_counts().compute()
    drop_idx = n_trips_by_veh.loc[n_trips_by_veh < params["min_trips_per_veh"]].index

    veh_cols = [params["veh_id_col"]] + params["veh_attr_cols"]
    vehs = trips.loc[:, veh_cols].drop_duplicates().compute()
    vehs = vehs.set_index(params["veh_id_col"]).sort_index()
    vehs = vehs.drop(index=drop_idx)
    return vehs


def mark_weight_class_group(vehs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Merge a VIUS-to-model weight-class correspondence onto the vehicle table.

    Args:
        vehs: Vehicle attributes DataFrame indexed by vehicle ID.
        params: Pipeline parameters dict with keys:

            - ``values`` (dict): correspondence mapping values.
            - ``id_dimensions`` (list[str]): columns used to join the
              correspondence table.
            - ``value_col`` (str): name of the output weight-class-group
              column.

    Returns:
        The vehicle DataFrame with a new weight-class-group column added.
    """
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


def get_vehicle_observation_frames(
    vehs: pd.DataFrame, dw: DwellSet, params: dict
) -> pd.DataFrame:
    """Record each vehicle's temporal span, total distance, and first/last location.

    Aggregates dwell-level records to one row per vehicle, capturing the
    observation window (first start time to last end time), total trip
    distance, and the hexagons at the start and end of the observation period.
    These values are used to weight vehicle-level statistics in downstream
    sampling.

    Args:
        vehs: Vehicle attributes DataFrame indexed by vehicle ID.
        dw: ``DwellSet`` containing the full dwell history; assumed to be
            sorted by (vehicle, time) or a Dask-backed instance (sorting
            is assumed in that case).
        params: Pipeline parameters dict with keys:

            - ``column_namer`` (dict[str, str]): mapping from generic aggregation
              output names to final column names (e.g., ``dist_traveled_col``).

    Returns:
        The vehicle DataFrame with observation-frame columns merged in:
        first/last timestamps, first/last hexagon, total trip distance, and
        total observation duration.
    """
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


def get_operating_radius(
    vehs: pd.DataFrame, dw: DwellSet, params: dict
) -> pd.DataFrame:
    """Compute the maximum haversine distance from any dwell to the vehicle's center.

    Projects the dwell GeoDataFrame to the H3 geographic CRS (WGS-84) so that
    ``calc_operating_radius`` can use haversine distances, then groups by
    vehicle and aggregates the geometry series.  For Dask-backed ``DwellSet``
    instances the computation is distributed with ``map_partitions`` and then
    triggered with a ``ProgressBar``.

    Args:
        vehs: Vehicle attributes GeoDataFrame indexed by vehicle ID.
        dw: Spatialised ``DwellSet`` containing H3-centroid geometries.
        params: Pipeline parameters dict with keys:

            - ``out_col`` (str): name of the output operating-radius column
              (in miles).

    Returns:
        The vehicle DataFrame with ``params["out_col"]`` added, containing
        the operating radius in miles for each vehicle.
    """
    dw.data = dw.data.to_crs(H3_CRS)  # since calc_operating_radius uses haversine dist

    def _get_op_rad(part: pd.DataFrame, grp_cols: list) -> pd.Series:
        return part.groupby(grp_cols).geometry.agg(calc_operating_radius).astype(float)

    kws = {
        "grp_cols": [dw.veh],
    }

    if dw.is_dask:
        meta_idx = pd.Index(np.array([], dtype=np.int64), name=dw.veh)
        meta = pd.Series([], name="radii", index=meta_idx)
        radii = dw.data.map_partitions(_get_op_rad, meta=meta, **kws)
        with ProgressBar():
            radii = radii.compute()
    else:
        radii = _get_op_rad(dw.data, **kws)

    vehs[params["out_col"]] = vehs.index.map(radii)
    return vehs


def get_time_weighted_centers(
    vehs: pd.DataFrame, dw: DwellSet, params: dict
) -> gpd.GeoDataFrame:
    """Compute the dwell-duration-weighted geographic center for each vehicle.

    Projects dwells to a planar CRS, computes dwell duration in hours, then
    calls ``find_time_weighted_centers`` to calculate a weighted centroid per
    vehicle.  The resulting centers are re-projected to the H3 geographic CRS
    before being merged onto the vehicle table, which is returned as a
    ``GeoDataFrame`` with the center column as the active geometry.

    Args:
        vehs: Vehicle attributes DataFrame indexed by vehicle ID.
        dw: Spatialised ``DwellSet``; geometries are projected in-place to
            ``params["proj_crs"]`` during this call.
        params: Pipeline parameters dict with keys:

            - ``proj_crs`` (str | CRS): projected CRS for planar distance
              calculations (e.g., an equal-area projection).
            - ``out_col`` (str): name of the output geometry column for the
              time-weighted center.

    Returns:
        A ``gpd.GeoDataFrame`` indexed by vehicle ID with a new point geometry
        column ``params["out_col"]`` containing each vehicle's time-weighted
        center in the H3 geographic CRS.
    """
    proj_crs = params["proj_crs"]
    dw.data = dw.data.to_crs(proj_crs)

    dw.data["dwell_hrs"] = total_hours(dw.data[dw.end] - dw.data[dw.start])

    ccol = "centers"
    kws = {
        "grp_col": dw.veh,
        "weight_col": "dwell_hrs",
        "center_col": ccol,
    }
    if dw.is_dask:
        meta_idx = pd.Index(np.array([], dtype=np.int64), name=dw.veh)
        meta = gpd.GeoSeries([], name=ccol, crs=proj_crs, index=meta_idx)
        meta = gpd.GeoDataFrame(meta, geometry=ccol)
        centers = dw.data.map_partitions(find_time_weighted_centers, **kws, meta=meta)
        with ProgressBar():
            centers = centers.compute()
    else:
        centers = find_time_weighted_centers(gdf=dw.data, **kws)

    centers = centers.to_crs(H3_CRS)
    vehs[params["out_col"]] = vehs.index.map(centers.loc[:, ccol])
    vehs = gpd.GeoDataFrame(data=vehs, geometry=params["out_col"])
    return vehs


def get_operating_segment(
    vehs: gpd.GeoDataFrame, dw: DwellSet, params: dict
) -> gpd.GeoDataFrame:
    """Assign each vehicle's primary operating-distance class from its dwell geography.

    For each dwell, computes the planar distance from the dwell's H3-centroid
    to the vehicle's time-weighted center (``params["center_col"]``).  Each
    distance is placed into a labelled bin defined by ``radius_bin_low_bounds_miles``.
    The cumulative trip mileage within each (vehicle, bin) combination is then
    summed, and the bin with the highest total mileage is selected as the primary
    operating-distance class.

    Vehicles without a home base use the time-weighted center as the reference
    point; since the center is already computed from all dwells, this is
    handled automatically by the geometry merge.

    Args:
        vehs: Vehicle GeoDataFrame with a geometry column for time-weighted
            centers (output of ``get_time_weighted_centers``).
        dw: Spatialised ``DwellSet``; geometries are projected in-place to
            ``params["proj_crs"]`` during this call.
        params: Pipeline parameters dict with keys:

            - ``proj_crs`` (str | CRS): projected CRS for planar distance
              measurements.
            - ``center_col`` (str): column name for the time-weighted center
              geometry in ``vehs``.
            - ``radius_bin_low_bounds_miles`` (dict[str, float]): ordered
              mapping of label → lower bound (miles) for each distance bin.
            - ``segment_col`` (str): name of the output operating-segment
              column.

    Returns:
        The vehicle GeoDataFrame with ``params["segment_col"]`` added,
        containing the primary operating-distance class label for each vehicle.
    """
    proj_crs = params["proj_crs"]
    dw.data = dw.data.to_crs(proj_crs)

    ccol = params["center_col"]
    centers: gpd.GeoDataFrame = vehs.loc[:, [ccol]]
    centers = centers.to_crs(proj_crs)
    dw.data = dw.data.merge(centers, how="left", on=dw.veh)
    dw.data["rad_miles"] = dw.data.geometry.distance(dw.data[ccol]) / METERS_PER_MILE
    max_rad = np.inf
    bins = params["radius_bin_low_bounds_miles"]
    labs = list(bins.keys())
    kws = {
        "bins": list(bins.values()) + [max_rad],
        "labels": labs,
        "include_lowest": True,
    }

    if dw.is_dask:

        def _categorize(part_ser, **kws):
            cat_ser = pd.cut(part_ser, **kws).astype(str)
            return cat_ser

        meta = pd.Series([], name="rad_miles_bin", dtype=str)
        dw.data["rad_miles_bin"] = dw.data["rad_miles"].map_partitions(
            _categorize, meta=meta, **kws
        )
    else:
        dw.data["rad_miles_bin"] = pd.cut(dw.data["rad_miles"], **kws)

    segs = dw.data.groupby([dw.veh, "rad_miles_bin"])[dw.trip_dist].sum()
    if dw.is_dask:
        with ProgressBar():
            segs = segs.compute()

    segs = segs.unstack(level="rad_miles_bin", fill_value=0.0)
    segs[params["segment_col"]] = segs.idxmax(axis=1)

    vehs = vehs.merge(segs.loc[:, params["segment_col"]], how="left", on=dw.veh)
    return vehs


### As of 11/19/2025, the following functions are no longer used, but could be useful
###     in the future.
def classify_vehicles(
    vehs: pd.DataFrame, veh_locs: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Classify vehicles by home-base status from a vehicle-location summary table.

    .. deprecated::
        Not connected to the active pipeline as of 2025-11-19.  Preserved for
        potential future use.

    Args:
        vehs: Vehicle attributes DataFrame.
        veh_locs: Vehicle-location-pair summary with a multi-index of
            (vehicle_id, hex_id).
        params: Pipeline parameters dict with keys:

            - ``veh_col`` (str): vehicle identifier column.
            - ``loc_col`` (str): location-type column.
            - ``base_location_type`` (str): location-type label that indicates
              a depot/home base.

    Returns:
        The vehicle DataFrame with a boolean ``has_home_base`` column added.
    """
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


def mark_vehicle_centers(
    vehs: pd.DataFrame, veh_locs: pd.DataFrame, params: dict, dwell_params: dict
) -> pd.DataFrame:
    """Assign a characteristic home-base hexagon to each vehicle with a known depot.

    Selects the highest-priority depot location for vehicles that have one,
    using sort columns and ascending flags from ``params``.  Vehicles without
    a depot receive a sentinel hexagon value (``params["nan_int"]``) cast to
    ``int`` so that all values share the same dtype.

    .. deprecated::
        Not connected to the active pipeline as of 2025-11-19.  Preserved for
        potential future use.

    Args:
        vehs: Vehicle attributes DataFrame.
        veh_locs: Vehicle-location-pair summary.
        params: Pipeline parameters dict with keys:

            - ``location_col`` (str): column containing the location-type
              label.
            - ``base_location_type`` (str): label used to identify depot
              locations.
            - ``sort_primary_locations_to_top`` (dict): ``columns`` and
              ``ascending`` lists for sorting.
            - ``home_base_col`` (str): output column name for the home-base
              hexagon.
            - ``nan_int`` (int): sentinel value for vehicles without a depot.
        dwell_params: ``DwellSet`` column-name parameters with keys ``veh``
            and ``hex``.

    Returns:
        The vehicle DataFrame indexed by ``veh_col`` with a home-base hexagon
        column added.
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


def mark_location_regions(
    vehs: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    params: dict,
) -> pd.DataFrame:
    """Assign a region label to each vehicle by spatial join of its home-base hexagon.

    Converts the home-base hexagon to a point geometry, reprojects to the
    region CRS, performs a left spatial join, and fills vehicles with no
    matched region (e.g., those without a home base) with a sentinel string.

    .. deprecated::
        Not connected to the active pipeline as of 2025-11-19.  Preserved for
        potential future use.

    Args:
        vehs: Vehicle DataFrame with a home-base hexagon column.
        regions: GeoDataFrame of region polygons.
        params: Pipeline parameters dict with keys:

            - ``veh_col`` (str): vehicle identifier column.
            - ``home_base_col`` (str): hexagon column for the home base.
            - ``nan_int`` (int): sentinel value indicating no home base.
            - ``region_name_col`` (str): column in ``regions`` containing the
              region label.
            - ``location_region_col`` (str): output column name.
            - ``na_region_fill`` (str): fill value for unmatched vehicles.

    Returns:
        The vehicle DataFrame with a region-label column added.
    """
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
