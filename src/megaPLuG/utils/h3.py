import geopandas as gpd
import h3.api.numpy_int as h3
import pandas as pd
from shapely import Polygon


def str_to_h3(s: pd.Series) -> pd.Series:
    return s.transform(h3.str_to_int)


def h3_to_poly(h: int) -> Polygon:
    bnd = h3.cell_to_boundary(h)
    # h3 outputs geometries in lat-lon format, but the convention in WGS84 is lon-lat
    bnd_flip = [(x, y) for y, x in bnd]
    poly = Polygon(bnd_flip)
    return poly


def cells_to_points(s: pd.Series) -> gpd.GeoSeries:
    """Creates a like-indexed GeoSeries of points from a series of h3 cells."""
    tups = s.transform(h3.cell_to_latlng)
    lats = tups.transform(lambda t: t[0]).values
    lngs = tups.transform(lambda t: t[1]).values
    pts = gpd.GeoSeries.from_xy(lngs, lats, crs="EPSG:4326", index=s.index)
    return pts


def cells_to_polygons(s: pd.Series) -> gpd.GeoSeries:
    """Creates a like-indexed GeoSeries of polygons from a series of h3 cells."""
    bnds = s.transform(h3.cell_to_boundary)
    flip_bnds = bnds.transform(lambda bnd: [(x, y) for y, x in bnd])
    polys = flip_bnds.transform(Polygon)
    poly_ser = gpd.GeoSeries(polys, crs="EPSG:4326", index=s.index)
    return poly_ser
