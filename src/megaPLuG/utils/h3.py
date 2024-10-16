from collections.abc import Callable

import dask.dataframe as dd
import dask_geopandas
import geopandas as gpd
import h3.api.numpy_int as h3
import pandas as pd
from shapely import Polygon

H3_CRS = "EPSG:4326"


def str_to_h3(s: pd.Series) -> pd.Series:
    return s.transform(h3.str_to_int)


def cells_to_points(s: pd.Series) -> gpd.GeoSeries:
    """Creates a like-indexed GeoSeries of points from a series of h3 cells."""
    tups = s.transform(h3.cell_to_latlng)
    lats = tups.transform(lambda t: t[0]).values
    lngs = tups.transform(lambda t: t[1]).values
    pts = gpd.GeoSeries.from_xy(lngs, lats, crs=H3_CRS, index=s.index)
    return pts


def cells_to_polygons(s: pd.Series) -> gpd.GeoSeries:
    """Creates a like-indexed GeoSeries of polygons from a series of h3 cells."""
    bnds = s.transform(h3.cell_to_boundary)
    flip_bnds = bnds.transform(lambda bnd: [(x, y) for y, x in bnd])
    polys = flip_bnds.transform(Polygon)
    poly_ser = gpd.GeoSeries(polys, crs=H3_CRS, index=s.index)
    return poly_ser


def add_geometries(
    data: pd.DataFrame | dd.DataFrame, hex_col: str, geom_type: str = "point"
) -> gpd.GeoDataFrame | dask_geopandas.GeoDataFrame:
    """Convert the underlying dataset into a GeoDataFrame."""
    if geom_type == "point":
        f = cells_to_points
    elif geom_type == "polygon":
        f = cells_to_polygons
    else:
        raise RuntimeError("Only 'point' and 'polygon' geometries are supported.")

    if isinstance(data, dd.DataFrame):
        data = dask_geopandas.from_dask_dataframe(df=data, geometry=None)
        data = data.map_partitions(_cells_to_geom_wrapper, f=f, hex_col=hex_col)
    elif isinstance(data, pd.DataFrame):
        data = gpd.GeoDataFrame(data=data, geometry=None)
        data = _cells_to_geom_wrapper(gdf=data, f=f, hex_col=hex_col)
    else:
        raise RuntimeError("Only Pandas and Dask dataframes supported.")
    return data


def _cells_to_geom_wrapper(
    gdf: gpd.GeoDataFrame, f: Callable[[pd.Series], gpd.GeoSeries], hex_col: str
) -> pd.DataFrame:
    """Convert a Pandas DataFrame to a GeoDataFrame using its hexagon column."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise RuntimeError("Incoming data is not a GeoDataFrame")

    if hex_col in gdf.columns:
        hexes = gdf[hex_col]
    elif hex_col in gdf.index.names:
        hexes = gdf.index.get_level_values(hex_col).to_series()
    else:
        raise RuntimeError(f"'{hex_col}' not found in DataFrame columns or index.")

    geoms = f(hexes)
    gdf = gdf.set_geometry(geoms)
    return gdf


def to_geospatial(df: pd.DataFrame, params: dict) -> gpd.GeoDataFrame:
    """Augment a pandas DataFrame with an H3 id column with the H3 geometries."""
    hexes = add_geometries(df, **params)
    return hexes
