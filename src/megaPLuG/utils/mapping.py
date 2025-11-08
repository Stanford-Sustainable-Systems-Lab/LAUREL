import geopandas as gpd
import h3.api.numpy_int as h3
import pandas as pd
from shapely import LineString

from megaplug.utils.h3 import cells_to_polygons


def map_dwells(
    df: pd.DataFrame, hue_col: str, hex_col: str, trajectories: bool = True, **kwargs
):
    """Plot a set of dwells using their columns."""
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
