"""
This is a boilerplate pipeline 'describe_locations'
generated using Kedro 0.19.3
"""

import logging

import geopandas as gpd

from megaPLuG.utils.h3 import region_polygons_to_cells

logger = logging.getLogger(__name__)

METERS_PER_MILE = 1609.344


def filter_urban_areas(urban: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Filter urban areas by their state code."""
    urban["state_usps"] = urban[params["area_name_col"]].str[-2:]
    urban_sel = urban.loc[urban["state_usps"].isin(params["state_codes"]), :]
    return urban_sel


def build_land_use_areas(
    govt: gpd.GeoDataFrame,
    highways: gpd.GeoDataFrame,
    urban: gpd.GeoDataFrame,
    params: dict,
) -> gpd.GeoDataFrame:
    """Build the urban, rural highway, and rural non-highway spatial divisions."""
    govt = govt.to_crs(params["crs"])
    highways = highways.to_crs(params["crs"])
    urban = urban.to_crs(params["crs"])

    buff_miles = params["highway_buffer_miles"]
    state_shp = govt.union_all()
    highways.geometry = highways.geometry.buffer(distance=buff_miles * METERS_PER_MILE)
    highways_shp = highways.union_all()
    urban_shp = urban.union_all()

    highways_final_shp = highways_shp.difference(urban_shp)
    non_rural = urban_shp.union(highways_shp)
    rural_final_shp = state_shp.difference(non_rural)

    land_use = gpd.GeoDataFrame().from_dict(
        data={
            params["land_use_col"]: ["urban", "rural", "highway"],
            "geometry": [urban_shp, rural_final_shp, highways_final_shp],
        },
        orient="columns",
        geometry="geometry",
        crs=params["crs"],
    )
    return land_use


def build_analysis_areas_node(
    govt: gpd.GeoDataFrame,
    infra: gpd.GeoDataFrame,
    land_use: gpd.GeoDataFrame,
    params: dict,
) -> gpd.GeoDataFrame:
    """Build the set of mutially-exclusive, collectively-exhaustive polygons which cover
    the study area.
    """
    return build_analysis_areas(infra, land_use, extent=govt, crs=params["crs"])


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
    """Get hexagon geometries for each study area.

    This implicitly assumes that the hexagons are small relative to the regions, so that
    there is a one-to-many relationship between regions and hexagons.
    """
    areas = areas.rename(
        columns={v: k for k, v in params["group_cols_renamer"].items()}
    )
    hexes = region_polygons_to_cells(
        geos=areas,
        grp_cols=list(params["group_cols_renamer"].keys()),
        hex_col=params["hex_col"],
    )
    hexes = hexes.reset_index()
    hexes = hexes.drop_duplicates(subset=params["hex_col"])
    hexes = hexes.set_index(params["hex_col"])
    return hexes
