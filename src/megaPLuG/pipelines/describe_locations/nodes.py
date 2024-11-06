"""
This is a boilerplate pipeline 'describe_locations'
generated using Kedro 0.19.3
"""

import logging

import geopandas as gpd

from megaPLuG.utils.h3 import region_polygons_to_cells

logger = logging.getLogger(__name__)


def build_analysis_areas(
    govt: gpd.GeoDataFrame,
    infra: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Build the set of mutually-exclusive, collectively-exhaustive polygons which
    cover the study area (e.g. state boundaries, utility territories).
    """
    govt = govt.to_crs(infra.crs)
    areas = govt.overlay(infra, how="intersection")
    return areas


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
