import logging

import geopandas as gpd
import pandas as pd

from megaPLuG.utils.h3 import cells_to_points

logger = logging.getLogger(__name__)


def build_utility_territory(infra: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Build a utility territory from a set of substation coordinates."""
    geo = infra.geometry.union_all().convex_hull.buffer(params["buffer_dist_meters"])
    utilities = pd.DataFrame.from_dict(
        {
            "utility": ["PG_and_E"],
            "territory": geo,
        }
    )
    utilities = gpd.GeoDataFrame(data=utilities, geometry="territory", crs=infra.crs)
    return utilities


def build_nearest_infra_corresp(
    hexes: pd.DataFrame,
    infra: gpd.GeoDataFrame,
    params: dict,
) -> pd.DataFrame:
    """Build a correspondence table between substations and charging locations."""
    orig_idx = hexes.index.names
    if orig_idx != [None]:
        hexes = hexes.reset_index()
    hexes = gpd.GeoDataFrame(hexes, geometry=cells_to_points(hexes[params["hex_col"]]))
    hexes = hexes.to_crs(infra.crs)
    renamer = params["substation_col_renamer"]
    infra_merge = infra.loc[:, list(renamer.values()) + [infra.geometry.name]]
    corresp = hexes.sjoin_nearest(infra_merge, how="left")
    corresp = corresp.drop(columns=["index_right", infra.geometry.name])
    corresp = corresp.rename(columns={v: k for k, v in renamer.items()})
    corresp = corresp.reset_index(drop=True)
    corresp = corresp.convert_dtypes()
    n_orig = len(corresp)
    corresp = corresp.drop_duplicates(subset=params["hex_col"])
    n_fin = len(corresp)
    if n_fin < n_orig:
        logger.warning(
            f"Dropped {n_orig - n_fin} duplicated values from the nearest infrastructure correspondence."
        )
    if orig_idx != [None]:
        corresp = corresp.set_index(orig_idx)
    return corresp
