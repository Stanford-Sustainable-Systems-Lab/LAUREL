"""Geometric utilities for vehicle operating-radius and time-weighted center calculations.

This module provides the spatial computation functions needed to characterise
each vehicle's spatial footprint: where it spends most of its dwell time
(time-weighted center) and how far it ranges from that center (operating
radius as half the diameter of the convex hull of its dwell locations).

Key functions:

- :func:`find_time_weighted_centers` — weighted centroid of projected dwell
  coordinates.
- :func:`calc_operating_radius` — convex-hull diameter via rotating calipers,
  using Haversine distances for geographic accuracy.
- :func:`calc_haversine_dist` and :func:`calc_max_dist_calipers` — Numba JIT
  inner kernels.

Key design decisions
--------------------
- **Projected coordinates required**: :func:`find_time_weighted_centers`
  requires the GeoDataFrame to be in a projected (metric) CRS so that the
  weighted mean of easting/northing coordinates is geometrically meaningful.
  The caller is responsible for reprojection.
- **Rotating calipers**: The maximum pairwise distance across a convex hull is
  computed in O(n) using rotating calipers rather than O(n²) brute-force.
  For degenerate cases (single point or line), the diameter is 0 or the direct
  Haversine distance respectively.
- **Numba JIT**: Both :func:`calc_haversine_dist` and
  :func:`calc_max_dist_calipers` are JIT-compiled for performance when called
  across thousands of vehicles.
"""

import geopandas as gpd
import numpy as np
from numba import jit
from numpy.typing import NDArray
from shapely.geometry import LineString, Point

EARTH_RADIUS_MILES = 3963.0
METERS_PER_MILE = 1609.344


def find_time_weighted_centers(
    gdf: gpd.GeoDataFrame, grp_col: str, weight_col: str, center_col: str = "centers"
) -> gpd.GeoDataFrame:
    """Compute the dwell-time-weighted geographic center for each vehicle.

    For each group identified by ``grp_col``, computes the weighted mean of the
    projected easting and northing coordinates, where the weight is
    ``weight_col`` (typically dwell duration).  Returns a GeoDataFrame of
    center points, one row per group.

    WARNING: This function may be very slow in Dask if the GeoDataFrame is not
    indexed on ``grp_col``.

    Args:
        gdf: Projected GeoDataFrame (metric CRS required) of vehicle dwells,
            with one row per dwell and a geometry column of point locations.
        grp_col: Column to group by (typically a vehicle ID).
        weight_col: Column of dwell durations or other non-negative weights.
        center_col: Name of the geometry column in the output GeoDataFrame.
            Defaults to ``"centers"``.

    Returns:
        GeoDataFrame indexed by ``grp_col`` with a single geometry column
        (``center_col``) of weighted centroid points in the same projected CRS.

    Raises:
        RuntimeError: If ``gdf`` has no CRS or its CRS is not projected.
    """
    if not hasattr(gdf, "crs"):
        raise RuntimeError("The DwellSet's underlying dataset is not geographic.")
    else:
        crs = gdf.crs
        if not crs.is_projected:
            raise RuntimeError(
                "The DwellSet's underlying dataset is not in projected coordinates."
            )

    gdf["easting_wt"] = gdf.geometry.x * gdf[weight_col]
    gdf["northing_wt"] = gdf.geometry.y * gdf[weight_col]
    centers = gdf.groupby(grp_col, sort=False).agg(
        {"easting_wt": "sum", "northing_wt": "sum", weight_col: "sum"}
    )
    gdf = gdf.drop(columns=["easting_wt", "northing_wt"])
    centers["easting"] = centers["easting_wt"] / centers[weight_col]
    centers["northing"] = centers["northing_wt"] / centers[weight_col]
    geoms = gpd.GeoSeries.from_xy(
        x=centers["easting"],
        y=centers["northing"],
        crs=crs,
        name=center_col,
        index=centers.index,
    )
    centers = gpd.GeoDataFrame(geoms, geometry=center_col)
    return centers


def calc_operating_radius(points: gpd.GeoSeries) -> float:
    """Estimate a vehicle's operating radius as half the diameter of its convex hull.

    The diameter is the maximum pairwise distance across the convex hull of all
    dwell locations, computed using the rotating-calipers algorithm
    (:func:`calc_max_dist_calipers`) for efficiency.  Haversine distances are
    used so the result is in miles regardless of the CRS of ``points``.

    Degenerate cases are handled explicitly:

    - A single point → radius 0.
    - Two points (a line) → half the direct Haversine distance.
    - A polygon → rotating-calipers diameter / 2.

    Args:
        points: GeoSeries of Shapely ``Point`` geometries (longitude, latitude)
            representing the vehicle's dwell locations.

    Returns:
        Operating radius in miles.
    """
    # Get convex hull points
    convex_hull = points.union_all().convex_hull
    if isinstance(convex_hull, Point):
        return 0.0
    elif isinstance(convex_hull, LineString):
        hull_lonlat = np.array(convex_hull.coords)
        diam = calc_haversine_dist(
            pt1=hull_lonlat[0], pt2=hull_lonlat[1], radius=EARTH_RADIUS_MILES
        )
    else:
        # Use rotating calipers to find the maximum distance
        hull_lonlat = np.array(convex_hull.exterior.coords)
        diam = calc_max_dist_calipers(
            hull_lonlat=hull_lonlat, radius=EARTH_RADIUS_MILES
        )
    rad = diam / 2
    return rad


@jit
def calc_max_dist_calipers(hull_lonlat: NDArray[np.floating], radius: float) -> float:
    """Calculate the maximum distance across a convex hull using rotating calipers."""
    n = hull_lonlat.shape[0]
    max_distance = 0.0
    j = 1  # Start with the second point on the hull
    for i in range(n):
        # Check the distance between point i and the farthest point on the hull
        while True:
            next_idx = (j + 1) % n
            cur_dist = calc_haversine_dist(
                hull_lonlat[i], hull_lonlat[j], radius=radius
            ).item()  # Item assumes that an NDArray is returned with one element only
            nex_dist = calc_haversine_dist(
                hull_lonlat[i], hull_lonlat[next_idx], radius=radius
            ).item()
            if nex_dist > cur_dist:
                j = next_idx  # Move the "caliper"
            else:
                break
        # Update the maximum distance found
        max_distance = np.maximum(max_distance, cur_dist)
    return max_distance


@jit
def calc_haversine_dist(
    pt1: NDArray[np.floating], pt2: NDArray[np.floating], radius: float
) -> NDArray[np.floating]:
    """Calculate the Haversine distance between two points on Earth's surface.

    Assumes (longitude, latitude) points. Each row is a point.
    """
    caster = np.ones((1, 1))  # Used to cast to a 2-D array
    pt1 = caster * np.radians(pt1)
    pt2 = caster * np.radians(pt2)
    diffs = pt2 - pt1

    numer = (
        1
        - np.cos(diffs[:, 1])
        + np.cos(pt1[:, 1]) * np.cos(pt2[:, 1]) * (1 - np.cos(diffs[:, 0]))
    )
    res = 2 * radius * np.arcsin(np.sqrt(numer / 2))
    return res
