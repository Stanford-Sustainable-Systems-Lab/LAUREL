"""Interactive map visualisation for vehicle dwell sequences.

Provides :func:`map_dwells`, a thin wrapper around ``geopandas.GeoDataFrame.explore``
that renders dwell locations as hexagon polygons and, optionally, draws
trajectory lines connecting consecutive dwell centroids.  Intended for use in
exploratory notebooks to inspect individual vehicle routing patterns.
"""

import geopandas as gpd
import h3.api.numpy_int as h3
import pandas as pd
from shapely import LineString

from laurel.utils.h3 import cells_to_polygons


def map_dwells(
    df: pd.DataFrame, hue_col: str, hex_col: str, trajectories: bool = True, **kwargs
):
    """Render dwell locations as an interactive map with optional trajectory lines.

    Converts H3 cell IDs to hexagon polygons, colours them by ``hue_col``, and
    displays the result via ``folium`` (through ``geopandas.explore``).  If
    ``trajectories=True``, a second layer of ``LineString`` segments is drawn
    connecting the representative point of each dwell to that of the next,
    giving a visual impression of the vehicle's route.

    Args:
        df: Time-ordered DataFrame of dwell observations for one or more
            vehicles, with at least ``hex_col`` and ``hue_col`` present.
        hue_col: Column used to colour hexagons (e.g. ``"charging_mode"``).
            Categorical columns are cast to ``str`` before plotting.
        hex_col: Column of H3 cell IDs (integer or string format both accepted).
        trajectories: If ``True`` (default), draw line segments between
            consecutive dwell centroids on a second map layer.
        **kwargs: Additional keyword arguments forwarded to
            ``GeoDataFrame.explore`` (e.g. ``cmap``, ``tooltip``, ``tiles``).

    Returns:
        A ``folium.Map`` object that can be displayed inline in a notebook.
    """
    if not pd.api.types.is_integer_dtype(df[hex_col]):
        hexes = df[hex_col].transform(h3.str_to_int)
    else:
        hexes = df[hex_col]
    gdf = gpd.GeoDataFrame(data=df, geometry=cells_to_polygons(hexes))
    # if pd.api.types.is_integer_dtype(df[hex_col]):
    #     df[hex_col] = df[hex_col].transform(h3.int_to_str)

    if isinstance(gdf[hue_col].dtype, pd.CategoricalDtype):
        gdf[hue_col] = gdf[hue_col].astype("str")
    m = gdf.explore(column=hue_col, **kwargs)

    if trajectories:
        gdf["centroid_prev"] = gdf.geometry.representative_point()
        gdf["centroid"] = gdf["centroid_prev"].shift(1)

        lines = []
        for i in range(len(gdf)):
            start, end = gdf["centroid_prev"].iloc[i], gdf["centroid"].iloc[i]
            if start is not None and end is not None:
                line = LineString([start, end])
            else:
                line = None
            lines.append(line)
        gdf["trajectory"] = gpd.GeoSeries(lines, crs=gdf.crs, index=gdf.index)
        gdf = gdf.drop(columns=["centroid_prev", "centroid"])

        gdf = gdf.set_geometry("trajectory")
        m = gdf.explore(m=m, tooltip=False)
    return m
