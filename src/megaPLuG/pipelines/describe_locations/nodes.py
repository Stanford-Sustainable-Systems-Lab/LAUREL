"""
This is a boilerplate pipeline 'describe_locations'
generated using Kedro 0.19.3
"""

import logging

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from megaPLuG.utils.h3 import (
    H3_CRS,
    H3_DEFAULT_RESOLUTION,
    coords_to_cells_wrapper,
    region_polygons_to_cells,
)

logger = logging.getLogger(__name__)

METERS_PER_MILE = 1609.344


def format_substation_boundaries_pg_and_e(
    infra: gpd.GeoDataFrame, params: dict
) -> gpd.GeoDataFrame:
    """Add up substation capacities from the capacities of transformer banks."""
    infra = infra.rename(columns={v: k for k, v in params["col_renamer"].items()})
    infra["substation_id"] = infra["substation_id"].astype(int)

    subs = infra.dissolve(
        by="substation_id",
        aggfunc={
            "substation_name": "first",
            "rating_mw": "sum",
        },
    )
    subs[params["add_state_col"]["name"]] = params["add_state_col"]["value"]
    return subs


def format_substation_profiles(profs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Collapse profiles from hour-month combinations to a single characteristic day."""
    pcols = params["columns"]
    profs = profs.rename(columns={v: k for k, v in params["col_renamer"].items()})

    split = profs[pcols["month_hour"]].str.split("_")
    profs[pcols["month"]] = split.transform(lambda x: int(x[0]))
    profs[pcols["hour"]] = split.transform(lambda x: int(x[1]))
    profs = profs.drop(columns=[pcols["month_hour"]])

    # Aggregate to a characteristic day
    profs = profs.groupby([pcols["substation_id"], pcols["hour"]]).agg(
        max_base_by_hour_kw=pd.NamedAgg(pcols["baseload"], "max"),
    )
    profs["max_base_by_hour_mw"] = profs["max_base_by_hour_kw"] / 1000
    profs["max_base_mw"] = profs.groupby(pcols["substation_id"])[
        "max_base_by_hour_mw"
    ].transform(lambda s: s.max())
    profs = profs.drop(columns=["max_base_by_hour_kw"])
    profs = profs.reset_index(pcols["hour"])
    return profs


def describe_substation_usage(
    profs: pd.DataFrame, subs: gpd.GeoDataFrame, params: dict
) -> pd.DataFrame:
    """Combine baseload profiles and capacities to describe substation usage."""
    pcols = params["columns"]
    subs = subs.drop(columns=params["drop_substation_cols"])
    subs = profs.merge(subs, how="inner", on=pcols["substation_id"])
    subs = subs.reset_index()
    subs = subs.sort_values([pcols["substation_id"], pcols["hour"]])
    subs = subs.set_index(pcols["substation_id"])
    subs[pcols["cap_avail_mw"]] = subs[pcols["rating_mw"]] - subs[pcols["baseload_mw"]]
    return subs


def format_substation_boundaries_contin(
    infra: gpd.GeoDataFrame, params: dict
) -> gpd.GeoDataFrame:
    """Add up substation capacities from the capacities of transformer banks."""
    infra = infra.rename(columns={v: k for k, v in params["col_renamer"].items()})
    infra["substation_id"] = infra["substation_id"].astype(int)
    orig_len = len(infra)
    infra = infra.dropna(subset=infra.geometry.name)
    new_len = len(infra)
    if new_len < orig_len:
        d = int(orig_len - new_len)
        logger.warning(f"{d} substations dropped because of missing geometries.")
    return infra


def build_substation_polygons(
    subs: gpd.GeoDataFrame, states: gpd.GeoDataFrame, params: dict
) -> gpd.GeoDataFrame:
    """Get substation polygons from points and state boundaries."""
    states = states.to_crs(params["proj_crs"])
    subs = subs.to_crs(params["proj_crs"])

    # Calculate Voronoi polygons, then merge them back onto the substation dataframe
    orig_geo_name = subs.geometry.name
    vor_polys = subs.geometry.voronoi_polygons()
    substs_polys = gpd.GeoDataFrame(geometry=vor_polys)
    substs_polys.rename_geometry("substation_polygons", inplace=True)
    subs_polys = subs.sjoin(substs_polys, how="left", predicate="intersects")
    subs_polys = subs_polys.merge(
        substs_polys, how="left", left_on="index_right", right_index=True
    )
    subs_polys.set_geometry("substation_polygons", inplace=True)

    # Clip the substation polygons at the bounds
    ddf = dgpd.from_geopandas(subs_polys, npartitions=params["n_partitions"])
    bounds = states.geometry.union_all()
    subs_polys[params["poly_col_out"]] = ddf.intersection(bounds).compute()
    subs_polys.set_geometry(params["poly_col_out"], inplace=True)
    subs_polys = subs_polys.drop(columns=[orig_geo_name, "substation_polygons"])
    return subs_polys


def format_govt_areas(states: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Format the states dataset."""
    states = states.rename(columns={v: k for k, v in params["col_renamer"].items()})
    states = states.loc[:, params["keep_cols"] + [states.geometry.name]]
    return states


def format_urban(urban: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Format the urban areas dataset."""
    urban[params["state_col"]] = urban[params["area_name_col"]].str[-2:]
    return urban


def format_highways(
    highways: gpd.GeoDataFrame, states: gpd.GeoDataFrame, params: dict
) -> gpd.GeoDataFrame:
    """Format the highways dataset."""
    scols = params["state_cols"]
    corresp = states.loc[:, [scols["name"], scols["code"]]]
    highways = highways.merge(
        corresp,
        how="inner",
        left_on=params["highway_cols"]["state"],
        right_on=scols["name"],
    )
    highways = highways.rename(columns={v: k for k, v in params["col_renamer"].items()})
    highways = highways.loc[:, params["keep_cols"] + [highways.geometry.name]]
    return highways


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
    out = extent.to_crs(crs)
    for gdf in tqdm(args):
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


def format_estabs(
    estabs_core: dd.DataFrame,
    estabs_geo: dd.DataFrame,
    estabs_rels: dd.DataFrame,
    params: dict,
) -> dgpd.GeoDataFrame:
    """Format the establishment dataset by merging together disparate raw datasets."""
    rnmr = {v: k for k, v in params["col_renamer"].items()}
    estabs_core = estabs_core.rename(columns=rnmr)
    estabs_geo = estabs_geo.rename(columns=rnmr)
    estabs_rels = estabs_rels.rename(columns=rnmr)

    estabs_core = estabs_core.set_index("estab_id")
    estabs_geo = estabs_geo.set_index("estab_id")
    estabs_rels = estabs_rels.set_index("estab_id")

    estabs_geo = estabs_geo.dropna(subset=["lon", "lat"])
    # estabs_geo = estabs_geo.categorize(columns=["state"])
    estabs_geo["geometry"] = dgpd.points_from_xy(
        df=estabs_geo, x="lon", y="lat", crs=H3_CRS
    )
    estabs_geo["hex_id"] = estabs_geo.map_partitions(
        coords_to_cells_wrapper,
        lat_col="lat",
        lng_col="lon",
        res=H3_DEFAULT_RESOLUTION,
        meta=("x", "uint64"),
    )
    estabs_geo = dgpd.from_dask_dataframe(df=estabs_geo, geometry="geometry")
    estabs_geo = estabs_geo.drop(columns=["lon", "lat"])

    estabs_core["naics_8"] = (
        estabs_core["naics_8"].fillna(params["default_naics"]).astype(int)
    )

    # Order matters here to preserve geometry-awareness
    estabs_mrg = estabs_geo.merge(
        estabs_core, how="inner", left_index=True, right_index=True
    )
    estabs_mrg = estabs_mrg.merge(
        estabs_rels, how="inner", left_index=True, right_index=True
    )

    out_cols = [col for col in list(rnmr.values()) if col in estabs_mrg.columns]
    out_cols.extend(params["calculated_keep_cols"])

    return estabs_mrg[out_cols]
