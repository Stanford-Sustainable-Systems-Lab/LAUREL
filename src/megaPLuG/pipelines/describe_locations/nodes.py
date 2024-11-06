"""
This is a boilerplate pipeline 'describe_locations'
generated using Kedro 0.19.3
"""

import logging

import geopandas as gpd

from megaPLuG.utils.h3 import region_polygons_to_cells

logger = logging.getLogger(__name__)


def build_analysis_areas_node(
    govt: gpd.GeoDataFrame,
    infra: gpd.GeoDataFrame,
    params: dict,
) -> gpd.GeoDataFrame:
    """Build the set of mutially-exclusive, collectively-exhaustive polygons which cover
    the study area.
    """
    return build_analysis_areas(infra, extent=govt, crs=params["crs"])


def build_analysis_areas(
    *args: list[gpd.GeoDataFrame], extent: gpd.GeoDataFrame, crs: str
) -> gpd.GeoDataFrame:
    """Build the set of mutually-exclusive, collectively-exhaustive polygons which
    cover the study area (e.g. state boundaries, utility territories).
    """
    # TODO: Extend this function to allow for 1) an extent shape and 2) multi-UTM crs
    out = extent.to_crs(crs)
    for gdf in args:
        gdf_proj = gdf.to_crs(crs)
        out = out.overlay(gdf_proj, how="intersection")
    return out


def get_hexes_by_area(areas: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Get hexagon geometries for each study area."""
    hexes = region_polygons_to_cells(
        geos=areas,
        grp_cols=params["group_cols"],
        hex_col=params["hex_col"],
    )
    hexes = hexes.reset_index()
    hexes = hexes.set_index(params["hex_col"])
    return hexes
