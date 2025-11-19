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
    """Find time-weighted center of a set of dwells.

    WARNING: This function may be very slow in Dask if the gdf is not indexed on grp_col
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
    geoms = gpd.GeoSeries.from_xy(x=centers["easting"], y=centers["northing"], crs=crs)
    geoms.name = center_col
    centers = gpd.GeoDataFrame(index=centers.index, geometry=geoms.values)
    return centers


def calc_operating_radius(points: gpd.GeoSeries) -> float:
    """Calculate the maximum pairwise distance using convex hull and rotating calipers."""
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
