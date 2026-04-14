"""H3 hexagonal grid utilities: coordinate conversion, geometry construction, and polygon-cell mapping.

This module provides the spatial primitives needed to work with Uber's H3
hierarchical geospatial index throughout LAUREL.  All H3 operations use
resolution 8 (cell diameter ≈ 0.46 km) and the WGS-84 geographic CRS
(``EPSG:4326``).

Key functions:

- Coordinate conversion: :func:`cells_to_points`, :func:`cells_to_polygons`,
  :func:`coords_to_cells`.
- GeoDataFrame construction: :func:`add_geometries` (handles both pandas and
  Dask inputs).
- Region operations: :func:`cells_to_poly` (union cells into a region shape),
  :func:`cells_to_region_polygons`, :func:`region_polygons_to_cells`.

Key design decisions
--------------------
- **Dask compatibility**: :func:`add_geometries` converts a Dask DataFrame to
  ``dask_geopandas`` via ``map_partitions`` so that geometry attachment can be
  parallelised; the CRS is set on the whole frame after partition mapping.
- **H3 integer API**: The module primarily uses ``h3.api.numpy_int`` (uint64
  cell IDs) for Numba/NumPy compatibility; string-based H3 APIs are imported
  only where needed (e.g. :func:`coords_to_cells` returns uint64).
"""

from __future__ import annotations

from collections.abc import Callable

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import h3.api.numpy_int as h3
import numpy as np
import pandas as pd
from shapely import Polygon

H3_CRS = "EPSG:4326"
H3_DEFAULT_RESOLUTION = 8


def str_to_h3(s: pd.Series) -> pd.Series:
    """Convert a Series of H3 hex-string cell IDs to uint64 integer IDs."""
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
) -> gpd.GeoDataFrame | dgpd.GeoDataFrame:
    """Attach H3 geometries to a DataFrame, returning a GeoDataFrame.

    Converts each H3 cell in ``hex_col`` to either a centroid point or the full
    hexagon polygon, then attaches the resulting ``GeoSeries`` as the geometry
    column.  Handles both pandas and Dask DataFrames; for Dask inputs the
    geometry is attached per-partition via ``map_partitions``.

    Args:
        data: pandas or Dask DataFrame with an H3 integer cell ID column.
        hex_col: Name of the column (or index level) holding H3 uint64 cell IDs.
        geom_type: ``"point"`` for centroid points or ``"polygon"`` for full
            hexagon polygons.

    Returns:
        GeoDataFrame (or Dask GeoDataFrame) with CRS ``EPSG:4326`` and a
        ``geometry`` column of Shapely objects.

    Raises:
        RuntimeError: If ``geom_type`` is neither ``"point"`` nor ``"polygon"``.
        RuntimeError: If ``data`` is neither a pandas nor a Dask DataFrame.
    """
    if geom_type == "point":
        f = cells_to_points
    elif geom_type == "polygon":
        f = cells_to_polygons
    else:
        raise RuntimeError("Only 'point' and 'polygon' geometries are supported.")

    if isinstance(data, dd.DataFrame):
        data = dgpd.from_dask_dataframe(df=data, geometry=None)
        meta = dd.utils.make_meta(data)
        meta_geo = gpd.GeoDataFrame(data=meta, geometry=gpd.GeoSeries(index=meta.index))
        data = data.map_partitions(
            _cells_to_geom_wrapper, f=f, hex_col=hex_col, meta=meta_geo
        )
        data = data.set_crs(H3_CRS)
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
    if "geometry" not in gdf.columns.names:
        geoms.name = "geometry"
    else:
        raise RuntimeError("'geometry' is already used as a column name.")
    gdf = gdf.set_geometry(geoms)
    return gdf


def to_geospatial(df: pd.DataFrame, params: dict) -> gpd.GeoDataFrame:
    """Kedro node wrapper: attach H3 geometries to a DataFrame.

    Args:
        df: pandas DataFrame with an H3 cell ID column.
        params: Keyword arguments forwarded to :func:`add_geometries`
            (``hex_col``, optionally ``geom_type``).

    Returns:
        GeoDataFrame with geometry column in ``EPSG:4326``.
    """
    hexes = add_geometries(df, **params)
    return hexes


def cells_to_poly(hser: pd.Series) -> h3.H3Shape:
    """Generate a region (multi-)polygon from a series of hexagon ids.

    This will often be useful in a groupby setting. For example:
        df.groupby("region_id")["hex_id"].agg(cells_to_poly)
    """
    geo = h3.cells_to_h3shape(hser.unique())
    return geo


def cells_to_region_polygons(
    corresp: pd.DataFrame, hex_col: str, region_col: str
) -> gpd.GeoDataFrame:
    """Build one (multi-)polygon per region from a hex→region correspondence table.

    Groups ``corresp`` by ``region_col``, unions the H3 cells in each group
    into a contiguous region shape via :func:`cells_to_poly`, and returns a
    GeoDataFrame indexed by region.

    Args:
        corresp: DataFrame with at least ``hex_col`` and ``region_col`` columns.
        hex_col: Name of the H3 uint64 cell ID column.
        region_col: Name of the region identifier column to group by.

    Returns:
        GeoDataFrame with columns ``[region_col, "geometry"]`` and CRS
        ``EPSG:4326``.
    """
    regions = corresp.groupby(region_col)[hex_col].agg(cells_to_poly)
    regions = gpd.GeoSeries(regions, name="geometry", crs=H3_CRS)
    regions = regions.reset_index()
    return regions


def region_polygons_to_cells(
    geos: gpd.GeoDataFrame, grp_cols: str | list[str], hex_col: str
) -> pd.DataFrame:
    """Explode a GeoDataFrame of (multi-)polygons to one row per H3 cell they cover.

    For each polygon in ``geos``, enumerates all H3 resolution-8 cells whose
    centroids fall within the polygon via ``h3.geo_to_cells``, then explodes the
    result to long form.  Empty or null geometries are silently dropped.

    Args:
        geos: GeoDataFrame of polygon or multipolygon geometries.
        grp_cols: Column name(s) identifying each region (carried through to
            the output).
        hex_col: Name of the output column for H3 uint64 cell IDs.

    Returns:
        Long DataFrame with columns ``grp_cols + [hex_col]`` and a default
        integer index.
    """
    if isinstance(grp_cols, str):
        grp_cols = [grp_cols]

    geo_col = geos.geometry.name
    geos = geos.to_crs(H3_CRS)

    if geos.empty:
        return pd.DataFrame(columns=grp_cols + [hex_col])

    geos = geos.dropna(subset=[geo_col]).copy()
    if geos.empty:
        return pd.DataFrame(columns=grp_cols + [hex_col])

    geos[hex_col] = geos[geo_col].apply(
        lambda geom: tuple(h3.geo_to_cells(geom, res=H3_DEFAULT_RESOLUTION))
    )
    hexes = geos.drop(columns=[geo_col])
    hexes = hexes.explode(hex_col)
    hexes = hexes.dropna(subset=[hex_col])
    if hexes.empty:
        return pd.DataFrame(columns=grp_cols + [hex_col])

    hexes[hex_col] = hexes[hex_col].astype(np.uint64)
    hexes = hexes.loc[:, grp_cols + [hex_col]]
    hexes = hexes.reset_index(drop=True)
    return hexes


def coords_to_cells(lat: np.ndarray, lng: np.ndarray, res: int) -> np.ndarray:
    """Convert parallel latitude/longitude arrays to H3 uint64 cell IDs at ``res``.

    Args:
        lat: 1-D array of latitudes in decimal degrees.
        lng: 1-D array of longitudes in decimal degrees (same length as ``lat``).
        res: H3 resolution (0–15).

    Returns:
        1-D ``uint64`` array of H3 cell IDs, same length as ``lat``.
    """
    assert lat.shape == lng.shape
    assert lat.ndim == 1
    assert lng.ndim == 1
    out = np.empty_like(lat, dtype=np.uint64)
    for i in range(out.size):
        out[i] = h3.latlng_to_cell(lat=lat[i], lng=lng[i], res=res)
    return out


def coords_to_cells_wrapper(
    part: pd.DataFrame, lat_col: str, lng_col: str, res: int
) -> np.ndarray:
    return coords_to_cells(lat=part[lat_col].values, lng=part[lng_col].values, res=res)
