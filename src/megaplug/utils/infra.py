"""Utilities for building substation-to-hex correspondence tables.

Provides two spatial helpers used in the describe_locations pipeline:

- :func:`build_utility_territory` — constructs a rough convex-hull territory
  polygon for a utility (currently PG&E) from its substation coordinates.
- :func:`build_nearest_infra_corresp` — assigns each H3 hexagon to its
  nearest substation via a spatial nearest-neighbor join, producing the
  hex→substation correspondence table used throughout the evaluate_impacts
  pipeline.
"""

import logging

import geopandas as gpd
import pandas as pd

from megaplug.utils.h3 import cells_to_points

logger = logging.getLogger(__name__)


def build_utility_territory(infra: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Build an approximate utility service-territory polygon from substation points.

    Unions all substation geometries, takes the convex hull, and buffers it by
    ``params["buffer_dist_meters"]`` to produce a single polygon representing the
    utility's rough service territory.  Currently hard-coded for PG&E.

    Args:
        infra: GeoDataFrame of substation point geometries in a projected CRS.
        params: Configuration dict with the following key:

            - **buffer_dist_meters** (``float``): Buffer distance in metres to
              apply around the convex hull.

    Returns:
        Single-row GeoDataFrame with columns ``["utility", "territory"]`` and the
        same CRS as ``infra``.
    """
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
    """Assign each H3 hexagon to its geographically nearest substation.

    Converts hex centroids to points (via :func:`cells_to_points`), reprojects
    to the infra CRS, performs a ``sjoin_nearest`` against the substation
    GeoDataFrame, and renames columns according to ``params["substation_col_renamer"]``.
    Duplicate hex assignments (which can arise when two substations are
    equidistant) are dropped with a warning.

    Args:
        hexes: DataFrame with an H3 cell ID column (``params["hex_col"]``) and
            optionally a named index.
        infra: GeoDataFrame of substation point geometries.
        params: Configuration dict with the following keys:

            - **hex_col** (``str``): Name of the H3 uint64 cell ID column in
              ``hexes``.
            - **substation_col_renamer** (``dict[str, str]``): Mapping from
              output column names to the corresponding column names in ``infra``
              (i.e. ``{desired_name: infra_col_name}``).

    Returns:
        DataFrame with one row per unique hex, containing the hex ID column plus
        the renamed substation attribute columns, indexed as in ``hexes``.
    """
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
