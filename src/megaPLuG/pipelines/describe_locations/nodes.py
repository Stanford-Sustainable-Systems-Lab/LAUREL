"""
This is a boilerplate pipeline 'describe_locations'
generated using Kedro 0.19.3
"""

import geopandas as gpd
import pandas as pd

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.h3 import add_geometries


def get_hex_geoms(dw: DwellSet, params: dict) -> gpd.GeoDataFrame:
    """Get hexagon geometries for each unique hexagon."""
    hexes = pd.DataFrame(dw.data[dw.hex].unique(), columns=[dw.hex])
    hexes = add_geometries(hexes, hex_col=dw.hex, geom_type=params["geom_type"])
    return hexes


def build_substation_location_corresp(
    hexes: gpd.GeoDataFrame,
    substs: gpd.GeoDataFrame,
    params: dict,
) -> pd.DataFrame:
    """Build a correspondence table between substations and charging locations."""
    territory = substs.geometry.unary_union.convex_hull.buffer(
        params["territory_buffer_meters"]
    )
    territory = gpd.GeoSeries(territory, crs=substs.crs)
    territory = territory.to_crs(hexes.crs)
    territory = territory.geometry[0]

    hexes_in = hexes.loc[hexes.geometry.within(territory)]
    hexes_in = hexes_in.to_crs(substs.crs)
    renamer = params["substation_col_renamer"]
    substs_merge = substs.loc[:, list(renamer.values()) + ["geometry"]]
    corresp = hexes_in.sjoin_nearest(substs_merge, how="left")
    corresp = corresp.drop(columns=["index_right", "geometry"])
    corresp = corresp.rename(columns={v: k for k, v in renamer.items()})
    corresp = corresp.reset_index(drop=True)
    corresp = corresp.convert_dtypes()
    return corresp
