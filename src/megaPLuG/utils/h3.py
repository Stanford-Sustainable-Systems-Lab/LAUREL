import geopandas as gpd
import h3.api.numpy_int as h3
import pandas as pd
from shapely import Polygon

H3_CRS = "EPSG:4326"


def str_to_h3(s: pd.Series) -> pd.Series:
    return s.transform(h3.string_to_h3)


def cells_to_points(s: pd.Series) -> gpd.GeoSeries:
    """Creates a like-indexed GeoSeries of points from a series of h3 cells."""
    tups = s.transform(h3.h3_to_geo)
    lats = tups.transform(lambda t: t[0]).values
    lngs = tups.transform(lambda t: t[1]).values
    pts = gpd.GeoSeries.from_xy(lngs, lats, crs=H3_CRS, index=s.index)
    return pts


def cells_to_polygons(s: pd.Series) -> gpd.GeoSeries:
    """Creates a like-indexed GeoSeries of polygons from a series of h3 cells."""
    bnds = s.transform(h3.h3_to_geo_boundary)
    flip_bnds = bnds.transform(lambda bnd: [(x, y) for y, x in bnd])
    polys = flip_bnds.transform(Polygon)
    poly_ser = gpd.GeoSeries(polys, crs=H3_CRS, index=s.index)
    return poly_ser
