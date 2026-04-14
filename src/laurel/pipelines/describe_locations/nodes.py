"""Kedro pipeline nodes for the ``describe_locations`` pipeline (Model Module 3 — Augment TAZs).

Constructs the spatial foundation of the model: H3 resolution-8 hexagonal
Traffic Analysis Zones (TAZs) covering the continental U.S., each labelled
with a freight-activity class that determines which charger types are deployed
there and how many.  This pipeline implements Model Module 3 (Augment TAZs)
from the paper.

Pipeline overview
-----------------
Substation territory construction:
1. **format_substation_boundaries_pg_and_e** — Sums transformer bank ratings
   to substation level and adds provenance metadata for PG&E ICA data.
2. **format_substation_profiles** — Collapses hour-month baseload profiles to
   a characteristic 24-hour day per substation.
3. **describe_substation_usage** — Joins profiles and capacities to compute
   available headroom at each substation hour.
4. **format_substations_contin** — Formats HIFLD point-based substation data
   (continental U.S.) for merging with polygon data.
5. **fill_out_substations** — Fills in substation territories not covered by
   ICA polygon data using Voronoi tessellation of HIFLD point locations.
6. **build_remainder_polys** — Constructs Voronoi-based territory polygons for
   point substations outside the ICA coverage area (called by
   ``fill_out_substations``).

Spatial layer formatting:
7. **format_states** — Renames columns in the U.S. state polygons layer.
8. **format_urban** — Renames columns in the urban-area polygons layer.
9. **format_highways** — Dissolves and buffers highway polylines to create
   highway-corridor polygons.
10. **build_land_use_areas** — Constructs three mutually exclusive land-use
    area polygons: urban, rural-highway, and rural-non-highway.
11. **clip_to_extent** — Clips any spatial layer to the extent of a reference
    layer.
12. **hexify_polygons** — Converts polygon layers to H3 hex grids by assigning
    each hex that overlaps the polygon the polygon's attributes.

Establishment data preparation:
13. **concat_columns** — Joins per-hexagon feature tables by shared hex index.
14. **fill_missingness** — Drops rows with missing required values and fills
    optional columns with configured defaults.
15. **prepare_shared_locations** — Formats shared truck-stop locations (e.g.,
    Jason's Law) with a standardised location-type label.
16. **format_estabs** — Merges raw Data Axle establishment core, geo, and
    relationship tables; assigns H3 hex IDs.
17. **reassign_hqs** — Reassigns NAICS codes for corporate-headquarters
    establishments to a HQ-specific code to prevent them from being
    misidentified as large freight facilities.
18. **prepare_stop_locations_public** — Formats publicly available truck-stop
    locations from Jason's Law for use as establishment records.
19. **get_osm_estabs_truck_stops** — Extracts fuel stations matching a truck-
    stop name pattern from an OSM PBF file.
20. **get_osm_estabs_warehouses** — Extracts warehouse/distribution-centre
    candidates from an OSM PBF file by name pattern.
21. **concat_extra_estabs** — Concatenates supplementary establishment sources.
22. **format_extra_estabs** — Reprojects extra establishments, assigns H3 hex
    IDs, and deduplicates.
23. **collapse_naics_classes** — Maps 8-digit NAICS codes to the smaller set
    of leaf classes used for clustering.
24. **pivot_hex_estabs** — Pivots the establishment table to a per-hexagon
    employment matrix (one column per NAICS leaf class).

NLCD land use raster extraction (``read_land_use`` tag):
25. **partition_hex_corresp** — Converts the pandas ``hex_base_corresp`` feather
    to a partitioned Dask parquet file for parallel raster extraction.
26. **read_land_use** — Extracts NLCD 2023 land cover fractions for every H3
    hexagon using ``exactextract``; writes ``hex_land_use`` in long format.

Freight-activity-class assignment:
27. **pivot_hex_land_use** — Pivots the land-use coverage table to a
    per-hexagon land-use-group fraction matrix.
28. **group_hexes** — Assigns each hexagon a freight-activity class using a
    combination of development threshold, freight-intensity rules, special-
    establishment flags, and K-Means clustering.
29. **apply_groups** — Merges freight-activity-class labels onto the hexagon
    feature table.

Key design decisions
--------------------
- **Voronoi gap-filling**: HIFLD substation data provides points for most of
  the U.S., but ICA data provides polygons for PG&E territory only.  Voronoi
  tessellation fills the remaining territory while avoiding overlap with known
  ICA polygons.  Disconnected Voronoi shards (islands separated from their
  substation point) are merged into the nearest connected territory to prevent
  orphaned hexagons.
- **Headquarters NAICS reassignment**: Data Axle assigns some large corporate
  headquarters the NAICS of the parent company's primary activity.  Without
  reassignment, a trucking-company HQ would be indistinguishable from a
  freight terminal.  The heuristic (employment ratio × business-status code ×
  establishment count) is conservative to avoid over-correction.
- **K-Means on sparse employment matrices**: The clustering uses log1p-
  transformed sparse COO matrices as input so that large employment counts in
  a single NAICS class do not dominate the distance metric.  Only hexagons
  with non-zero freight-intensive NAICS employment (excluding special
  categories like truck stops) are clustered; others receive rule-based labels.
- **Neighbor embedding**: Each hexagon's feature vector is augmented with the
  summed employment of its six H3 ring-1 neighbors to capture the local
  land-use context, which improves cluster coherence at the cluster-boundary
  hexagons.
- **Land use raster extraction runtime**: Extracting NLCD fractions for all H3
  resolution 8 hexagons in the Continental U.S. via ``exactextract`` is I/O-intensive
  and may take several hours on a single machine.  The step is isolated under
  the ``read_land_use`` tag so it can be run once and its output
  (``hex_land_use``) reused for subsequent runs.  The raster path is stored as
  a parameter rather than a catalog dataset because ``exactextract`` reads the
  GeoTIFF directly from disk and there is no standard Kedro dataset type that
  loads it in the required format.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.

Uber Technologies. H3: Hierarchical Hexagonal Geospatial Indexing System.
https://h3geo.org/

Data Axle USA. Business listings database.
U.S. DOT FHWA. Jason's Law Truck Parking Survey.
Homeland Infrastructure Foundation-Level Data (HIFLD). Electric substations.
USGS National Land Cover Database (NLCD).
"""

from __future__ import annotations

import logging
from pathlib import Path

import dask.dataframe as dd
import dask_geopandas as dgpd
import exactextract as ex
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from dask.dataframe.dispatch import make_meta
from dask.diagnostics.progress import ProgressBar
from dask.distributed import Client
from osmium.filter import KeyFilter, TagFilter
from sklearn.cluster import KMeans

from laurel.utils.h3 import (
    H3_CRS,
    H3_DEFAULT_RESOLUTION,
    add_geometries,
    coords_to_cells,
    coords_to_cells_wrapper,
    region_polygons_to_cells,
)
from laurel.utils.hex_neighbors import get_neighbor_embeddings
from laurel.utils.naics import get_naics_leaf_class
from laurel.utils.open_street_map import RegexTagFilter, get_gdf_from_filtered_osm

logger = logging.getLogger(__name__)

METERS_PER_MILE = 1609.344


def format_substation_boundaries_pg_and_e(
    infra: gpd.GeoDataFrame, params: dict
) -> gpd.GeoDataFrame:
    """Aggregate transformer bank ratings to substation level for PG&E ICA data.

    The PG&E ICA dataset provides one polygon per transformer bank; this
    function dissolves them to one polygon per substation, summing the MW
    ratings, and adds source and state metadata columns.

    Args:
        infra: Raw PG&E ICA GeoDataFrame with one row per transformer bank.
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names (e.g., renaming the capacity and ID columns).
            - ``add_state_col`` (dict): ``name`` and ``value`` for a static
              state-name column to add.
            - ``add_source_col`` (dict): ``name`` and ``value`` for a static
              data-source column to add.

    Returns:
        A ``gpd.GeoDataFrame`` indexed by ``substation_id`` with dissolved
        territory polygons, summed ratings, and metadata columns.
    """
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
    subs[params["add_source_col"]["name"]] = params["add_source_col"]["value"]
    return subs


def format_substation_profiles(profs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Collapse hourly-by-month PG&E baseload profiles to a characteristic 24-hour day.

    The raw PG&E ICA data encodes load as a ``month_hour`` string (e.g.,
    ``"1_14"`` for January hour 14).  This function splits the combined column,
    aggregates to the maximum baseload for each (substation, hour) combination
    across all months, and derives the substation's peak baseload across all
    hours.

    Args:
        profs: Raw PG&E ICA baseload profile DataFrame with a combined
            ``month_hour`` string column.
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names.
            - ``columns`` (dict): sub-keys ``month_hour``, ``month``, ``hour``,
              ``substation_id``, ``baseload`` (kW column name).

    Returns:
        A ``pd.DataFrame`` indexed by ``substation_id`` with columns
        ``hour``, ``max_base_by_hour_mw`` (max baseload for that hour, in MW),
        and ``max_base_mw`` (peak baseload across all hours, in MW).
    """
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
    """Join baseload profiles and rated capacities to compute hourly available headroom.

    Merges the characteristic-day profile onto the substation capacity table
    and computes ``cap_avail_mw = rating_mw - baseload_mw`` for each hour.

    Args:
        profs: Characteristic-day baseload profiles (output of
            ``format_substation_profiles``).
        subs: Substation GeoDataFrame with rated capacity.
        params: Pipeline parameters dict with keys:

            - ``drop_substation_cols`` (list[str]): columns to drop from
              ``subs`` before merging.
            - ``columns`` (dict): sub-keys ``substation_id``, ``hour``,
              ``rating_mw``, ``baseload_mw``, ``cap_avail_mw``.

    Returns:
        A ``pd.DataFrame`` indexed by ``substation_id``, sorted by hour,
        with columns for baseload, capacity, and available headroom (all MW).
    """
    pcols = params["columns"]
    subs = subs.drop(columns=params["drop_substation_cols"])
    subs = profs.merge(subs, how="inner", on=pcols["substation_id"])
    subs = subs.reset_index()
    subs = subs.sort_values([pcols["substation_id"], pcols["hour"]])
    subs = subs.set_index(pcols["substation_id"])
    subs[pcols["cap_avail_mw"]] = subs[pcols["rating_mw"]] - subs[pcols["baseload_mw"]]
    return subs


def format_substations_contin(subs: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Standardise the HIFLD continental substation GeoDataFrame for territory construction.

    Args:
        subs: Raw HIFLD substation point GeoDataFrame.
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names.
            - ``add_source_col`` (dict): ``name`` and ``value`` for a static
              data-source column.
            - ``keep_cols`` (list[str]): columns to retain.

    Returns:
        A standardised ``gpd.GeoDataFrame`` with integer ``substation_id``,
        a source label, and only the required columns retained.
    """
    subs = subs.rename(columns={v: k for k, v in params["col_renamer"].items()})
    subs["substation_id"] = subs["substation_id"].astype(int)
    subs[params["add_source_col"]["name"]] = params["add_source_col"]["value"]
    subs = subs.loc[:, params["keep_cols"]]
    return subs


def fill_out_substations(
    poly_subs: gpd.GeoDataFrame, point_subs: gpd.GeoDataFrame, params: dict
) -> gpd.GeoDataFrame:
    """Build complete substation territory coverage by supplementing ICA polygons with Voronoi polygons.

    Calls ``build_remainder_polys`` to create Voronoi-based territory polygons
    for HIFLD point substations that fall outside the PG&E ICA polygon
    coverage area, then concatenates both polygon sets into a single unified
    GeoDataFrame.

    Args:
        poly_subs: ICA polygon substations (output of
            ``format_substation_boundaries_pg_and_e``).
        point_subs: HIFLD point substations (output of
            ``format_substations_contin``).
        params: Pipeline parameters dict with keys:

            - ``columns`` (dict): sub-keys ``substation_id`` and ``source``.
            - ``proj_crs`` (str | CRS): projected CRS for distance operations.
            - ``buff_dist_meters`` (float): buffer distance used to compact
              disconnected Voronoi shards.

    Returns:
        A ``gpd.GeoDataFrame`` covering all substation territories, with a
        composite ``{substation_id}_{source}`` column indexing each territory.
    """
    pcols = params["columns"]
    sub_id_col = pcols["substation_id"]
    source_col = pcols["source"]

    point_subs_proj = point_subs.to_crs(params["proj_crs"])
    poly_subs_proj = poly_subs.to_crs(params["proj_crs"])

    rem_subs = build_remainder_polys(
        points=point_subs_proj,
        poly_mask=poly_subs_proj.geometry,
        buff_dist=params["buff_dist_meters"],
        id_col=sub_id_col,
    )

    sub_ls = [poly_subs_proj, rem_subs]
    renamer = {sub_id_col: f"{sub_id_col}_{source_col}"}
    sub_ls = [gdf.rename(columns=renamer) for gdf in sub_ls]
    all_subs = pd.concat(sub_ls, axis=0, ignore_index=True, join="inner")
    all_subs.index.name = sub_id_col
    all_subs = all_subs.reset_index()

    return all_subs


def build_remainder_polys(  # noqa: PLR0915
    points: gpd.GeoDataFrame, poly_mask: gpd.GeoSeries, buff_dist: float, id_col: str
) -> gpd.GeoDataFrame:
    """Construct Voronoi-based territory polygons for point substations outside the ICA coverage area.

    This function fills in the substation territory map for all point
    substations that are not already covered by a polygon from ``poly_mask``.
    The algorithm proceeds in seven stages:

    1. **Filter**: Retain only point substations whose location does not
       intersect the union of ``poly_mask`` polygons.
    2. **Voronoi**: Compute Voronoi polygons for the remaining point locations.
    3. **Difference**: Subtract the ``poly_mask`` union from each Voronoi
       polygon so that known ICA territories take precedence.
    4. **Explode**: Split multi-part polygons into individual shards and flag
       shards that do not contain their originating substation point as
       "disconnected" (these are Voronoi islands).
    5. **Compact donors**: Buffer disconnected shards by ``buff_dist`` and
       dissolve nearby ones together to reduce the number of donor groups.
    6. **Find acceptors**: For each compacted donor group, find the nearest
       connected ("acceptor") territory by spatial join and centroid proximity.
    7. **Dissolve**: Merge each donor shard into its chosen acceptor territory.

    Args:
        points: GeoDataFrame of point-substation locations; must be in the
            same CRS as ``poly_mask``.
        poly_mask: GeoSeries of existing territory polygons to subtract from
            the Voronoi result.
        buff_dist: Buffer distance (in the projected CRS units, typically
            metres) used to compact nearby disconnected shards.
        id_col: Column name in ``points`` containing the unique substation ID.

    Returns:
        A ``gpd.GeoDataFrame`` with one row per substation territory, matching
        the column schema of ``points``.

    Raises:
        ValueError: If ``points.crs`` does not match ``poly_mask.crs``.
    """
    if points.crs != poly_mask.crs:
        raise ValueError("Coordinate reference systems must be the same.")

    orig_col_names = points.columns
    orig_pt_geom_name = points.geometry.name
    poly_shp = poly_mask.union_all()

    # Filtering only point-based substations which are not covered by known substation polygons
    rem_pts = points.loc[~points.geometry.intersects(poly_shp)]

    # Building Vornoi polygons
    rem_pts.geometry.name = "location"
    vor_polys = rem_pts.geometry.voronoi_polygons()
    substs_polys = gpd.GeoDataFrame(geometry=vor_polys)
    substs_polys.rename_geometry("territory", inplace=True)
    subs_polys = rem_pts.sjoin(substs_polys, how="left", predicate="intersects")
    subs_polys = subs_polys.merge(
        substs_polys, how="left", left_on="index_right", right_index=True
    )
    subs_polys = subs_polys.rename_geometry("location")

    # Differencing Voronoi polygons against the known substation polygons
    coll = subs_polys.copy()
    coll = coll.set_index(id_col)
    coll = coll.set_geometry("territory")
    coll.geometry = coll.geometry.difference(poly_shp)

    # Mark the substation shards whose territory polygon does not contain the given substation location
    coll = coll.explode(index_parts=True)
    names_replace = list(coll.index.names)
    names_replace[-1] = "shard_id"
    coll.index.names = names_replace
    coll.loc[:, "disconnected"] = ~coll["territory"].contains(coll["location"])
    coll = coll.drop(columns=["index_right"])

    # Generate a new shard index
    joint_idx = coll.index.map(lambda idx: f"{idx[0]}_{idx[1]}")
    joint_idx.name = "donor_id"
    coll = coll.reset_index()
    coll.index = joint_idx
    coll = coll.reset_index()

    # Get donors (disconnected shards) and acceptors (connected shards)
    coll = coll.set_geometry("territory")
    donors = coll.loc[coll["disconnected"], ["donor_id", coll.geometry.name]]
    donors = donors.rename_geometry("donor_territory")

    acceptors = coll.loc[~coll["disconnected"], ["donor_id", coll.geometry.name]]
    acceptors = acceptors.rename(columns={"donor_id": "acceptor_id"})
    acceptors = acceptors.rename_geometry("acceptor_territory")

    # Compact donors together if they are within a distance of each other
    donors_compact = donors.copy()
    donors_compact["donor_territory_buffered"] = donors_compact[
        "donor_territory"
    ].buffer(distance=buff_dist)
    donors_compact = donors_compact.set_geometry("donor_territory_buffered")
    donors_compact = donors_compact.dissolve().explode().reset_index(drop=True)
    donors_compact = donors_compact.drop(columns=["donor_id"])
    donors_compact.index.name = "donor_compact_id"
    donors_compact = donors_compact.reset_index()
    donors_compact["donor_compact_id"] = donors_compact["donor_compact_id"].astype(str)
    donors_to_comp_donors = donors_compact.sjoin(
        donors, how="left", predicate="intersects"
    )
    donors_to_comp_donors = donors_to_comp_donors.loc[
        :, ["donor_compact_id", "donor_id"]
    ]

    donors = donors.merge(donors_to_comp_donors, how="left", on="donor_id")
    donors["area"] = donors.geometry.area
    donors = donors.sort_values(by=["area"], ascending=False)
    donors_small = donors.dissolve(by="donor_compact_id", aggfunc="first")
    donors_small = donors_small.drop(columns=["area", "donor_id"])
    donors_small = donors_small.reset_index()

    # Find the acceptors which touch each compacted donor area
    donors_small["donor_territory_buffered"] = donors_small["donor_territory"].buffer(
        distance=buff_dist
    )
    donors_small = donors_small.set_geometry("donor_territory_buffered")
    intersects = donors_small.sjoin(acceptors, how="left", predicate="intersects")
    intersects = intersects.drop(columns=["index_right"])
    # intersects = intersects.set_geometry("donor_territory")
    intersects = intersects.rename(columns={"acceptor_id": "acceptor_id_intersects"})

    # Find the touching acceptor's centroid which is closest to each donor polygon
    near_acceptors = acceptors.loc[
        acceptors["acceptor_id"].isin(intersects["acceptor_id_intersects"])
    ]
    near_acceptors.loc[:, "acceptor_centroid"] = near_acceptors.geometry.centroid
    near_acceptors = near_acceptors.set_geometry("acceptor_centroid")
    near_acceptors = near_acceptors.drop(columns=["acceptor_territory"])
    nearest = intersects.sjoin_nearest(near_acceptors, how="left")
    nearest = nearest.rename(columns={"acceptor_id": "acceptor_id_nearest"})

    # Select a single acceptor for each donor, the (centroid-)nearest touching acceptor
    nearest_touching = (
        nearest["acceptor_id_nearest"] == nearest["acceptor_id_intersects"]
    )
    donors_to_acceptors = nearest.loc[
        nearest_touching, ["donor_compact_id", "acceptor_id_nearest"]
    ]
    donors_to_acceptors = donors_to_acceptors.rename(
        columns={"acceptor_id_nearest": "acceptor_id"}
    )

    # Connect original donor_id to final acceptor_id through the two stages
    two_stager = donors_to_comp_donors.merge(
        donors_to_acceptors, how="left", on="donor_compact_id"
    )

    # Assign the chosen acceptor to each of the original shards.
    grouped = coll.merge(two_stager, how="left", on="donor_id")
    grouped.loc[:, "donor_compact_id"] = grouped["donor_id"].where(
        grouped["donor_compact_id"].isna(), grouped["donor_compact_id"]
    )
    grouped.loc[:, "acceptor_id"] = grouped["donor_id"].where(
        grouped["acceptor_id"].isna(), grouped["acceptor_id"]
    )

    # Dissolve to compact donors
    grouped["area"] = grouped["territory"].area
    grouped = grouped.sort_values(
        by=["donor_compact_id", "area"], ascending=[True, False]
    )
    diss_1 = grouped.dissolve(by="donor_compact_id", aggfunc="first")

    # Dissolve to connect donors with acceptors
    diss_1 = diss_1.sort_values(
        by=["acceptor_id", "disconnected"], ascending=[True, True]
    )
    diss_2 = diss_1.dissolve(by="acceptor_id", aggfunc="first")
    diss_2["location"].crs = diss_2["territory"].crs  # Resetting lost crs

    # Drop temporary columns
    dissolved = diss_2.drop(
        columns=["donor_id", "shard_id", "disconnected", "area", "location"]
    )
    dissolved = dissolved.rename_geometry(orig_pt_geom_name)
    dissolved = dissolved.reset_index(drop=True)
    dissolved = dissolved.loc[:, orig_col_names]

    return dissolved


def format_states(states: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Rename columns in the U.S. state polygons layer to internal names.

    Args:
        states: Raw state-boundary GeoDataFrame.
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names.

    Returns:
        The GeoDataFrame with columns renamed.
    """
    states = states.rename(columns={v: k for k, v in params["col_renamer"].items()})
    return states


def format_urban(urban: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Rename columns in the urban-areas polygons layer to internal names.

    Args:
        urban: Raw Census urban-area GeoDataFrame.
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names.

    Returns:
        The GeoDataFrame with columns renamed.
    """
    urban = urban.rename(columns={v: k for k, v in params["col_renamer"].items()})
    return urban


def format_highways(highways: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Dissolve and buffer highway polylines into corridor polygons.

    Dissolves all highway segments sharing the same ``dissolve_cols`` values
    into a single geometry, reprojects to a projected CRS for accurate
    buffering, applies a mile-radius buffer, then reprojects back to the
    original CRS.

    Args:
        highways: Raw highway polyline GeoDataFrame.
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names.
            - ``dissolve_cols`` (list[str]): columns to dissolve on.
            - ``highway_buffer_miles`` (float): buffer radius in miles.
            - ``buff_crs`` (str | CRS): projected CRS for buffering.

    Returns:
        A ``gpd.GeoDataFrame`` of highway-corridor polygons in the original
        CRS.
    """
    highways = highways.rename(columns={v: k for k, v in params["col_renamer"].items()})

    logger.info("Dissolving highways")
    highways = highways.dissolve(by=params["dissolve_cols"])
    highways = highways.reset_index()

    buff_miles = params["highway_buffer_miles"]
    orig_crs = highways.crs
    highways = highways.to_crs(params["buff_crs"])
    logger.info("Buffering highways")
    highways.geometry = highways.geometry.buffer(distance=buff_miles * METERS_PER_MILE)
    highways = highways.to_crs(orig_crs)
    return highways


def build_land_use_areas(
    govt: gpd.GeoDataFrame,
    highways: gpd.GeoDataFrame,
    urban: gpd.GeoDataFrame,
    params: dict,
) -> gpd.GeoDataFrame:
    """Construct three mutually exclusive land-use area polygons: urban, highway, and rural.

    The three areas are:

    - **urban**: the union of all Census urban-area polygons.
    - **highway**: the highway-corridor polygon union, minus the urban areas.
    - **rural**: the full state extent minus both urban and highway areas.

    Args:
        govt: State boundary GeoDataFrame used to define the full analysis
            extent.
        highways: Highway-corridor polygons (output of ``format_highways``).
        urban: Urban-area polygons (output of ``format_urban``).
        params: Pipeline parameters dict with keys:

            - ``crs`` (str | CRS): common CRS for all overlay operations.
            - ``highway_buffer_miles`` (float): additional buffer applied to
              the already-buffered highway polygons (set to 0 if not needed).
            - ``land_use_col`` (str): name of the land-use category column in
              the output.

    Returns:
        A ``gpd.GeoDataFrame`` with three rows (urban, highway, rural) and a
        ``land_use_col`` column, in ``params["crs"]``.
    """
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


def clip_to_extent(
    gdf: gpd.GeoDataFrame, extent: gpd.GeoDataFrame, params: dict
) -> pd.DataFrame:
    """Clip a geometry layer to the extent of a reference layer via polygon intersection.

    Reprojects both layers to ``params["crs"]``, performs an overlay
    intersection, drops empty or null geometries, and dissolves on all
    non-geometry attribute columns to merge adjacent fragments that share the
    same attributes.

    Args:
        gdf: Input GeoDataFrame to clip.
        extent: Reference GeoDataFrame whose union defines the clip boundary.
        params: Pipeline parameters dict with keys:

            - ``crs`` (str | CRS): CRS for the overlay operation.

    Returns:
        A ``pd.DataFrame`` (GeoDataFrame) clipped to the extent and dissolved
        on all attribute columns.
    """
    extent_proj = extent.loc[:, [extent.geometry.name]].to_crs(params["crs"])
    gdf_proj = gdf.to_crs(params["crs"])
    geo_clip = gdf_proj.overlay(extent_proj, how="intersection", make_valid=True)
    geo_clip = geo_clip.loc[~geo_clip.geometry.is_empty, gdf_proj.columns]
    geo_clip = geo_clip.dropna(subset=[geo_clip.geometry.name])

    geo_col = gdf_proj.geometry.name
    attr_cols = [col for col in gdf_proj.columns if col != geo_col]
    geo_compact = geo_clip.dissolve(by=attr_cols).reset_index()
    return geo_compact


def hexify_polygons(gdf: gpd.GeoDataFrame, params: dict) -> pd.DataFrame:
    """Convert polygon geometries to an H3 hexagon index by assigning hex IDs to overlapping hexes.

    For each polygon in ``gdf``, all H3 resolution-8 hexagons that overlap
    the polygon are identified and assigned the polygon's attribute values.
    When ``n_partitions > 1``, the operation is parallelised via Dask
    GeoDataFrame; otherwise it runs in-process.

    Args:
        gdf: GeoDataFrame of polygons, each with attribute columns to carry
            forward to the hex index.
        params: Pipeline parameters dict with keys:

            - ``hex_col`` (str): name of the H3 hex-ID output column.
            - ``n_partitions`` (int): number of Dask partitions (1 = no Dask).

    Returns:
        A ``pd.DataFrame`` indexed by H3 hex ID (``uint64``), with one row
        per unique hexagon and the polygon attribute columns attached.
        Duplicate hexagons (where a hexagon intersects multiple polygons) are
        dropped, keeping the first occurrence.
    """

    hex_col = params["hex_col"]
    kws = {
        "grp_cols": [col for col in gdf.columns if col != gdf.geometry.name],
        "hex_col": hex_col,
    }

    n_parts = params["n_partitions"]
    if n_parts > 1:
        geo_dask = dgpd.from_geopandas(data=gdf, npartitions=n_parts)
        meta_dict = {col: gdf[col].dtype for col in kws["grp_cols"]}
        meta_dict[hex_col] = np.uint64
        meta = make_meta(meta_dict)
        hex_dask = geo_dask.map_partitions(region_polygons_to_cells, **kws, meta=meta)
        with ProgressBar():
            hexes = hex_dask.compute()

    else:
        hexes = region_polygons_to_cells(gdf, **kws)

    logger.info("Drop duplicates and set index")
    hexes = hexes.reset_index(drop=True)
    hexes = hexes.drop_duplicates(subset=hex_col)
    hexes = hexes.set_index(hex_col)

    return hexes


def concat_columns(*args: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join multiple per-hexagon DataFrames on their shared hex index.

    Args:
        *args: Two or more ``pd.DataFrame`` objects sharing the same hex-ID
            index.

    Returns:
        A single ``pd.DataFrame`` with all columns from all inputs, joined
        on the shared index with ``join="outer"`` (hexagons present in any
        input are retained).
    """
    cat = pd.concat(args, axis=1, join="outer")
    return cat


def fill_missingness(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Drop rows with required missing values and fill optional missing values.

    Applies two sequential operations: first drops all rows with NaN in
    ``drop_na_cols`` (required columns); then fills NaN in ``fill_na_vals``
    columns with configured constants.  Handles ``CategoricalDtype`` columns
    correctly by adding the fill value to the category list before filling.

    Args:
        df: Input DataFrame, typically the concatenated hexagon feature table.
        params: Pipeline parameters dict with keys:

            - ``drop_na_cols`` (list[str] | None): columns whose NaN rows
              should be dropped entirely.
            - ``fill_na_vals`` (dict[str, any] | None): mapping from column
              name to fill value for optional NaN imputation.

    Returns:
        A ``pd.DataFrame`` with required NaN rows removed and optional NaN
        values filled.
    """
    # For some columns, drop all NA values
    dropna_cols = params.get("drop_na_cols", None)
    if dropna_cols is not None:
        df_filt = df.dropna(subset=dropna_cols).copy()
    else:
        df_filt = df.copy()

    # For other columns, fill NA values with chosen values
    fillna_vals = params.get("fill_na_vals", None)
    if fillna_vals is not None:
        for col, val in fillna_vals.items():
            col_data = df_filt[col]
            if isinstance(col_data.dtype, pd.CategoricalDtype):
                if val not in col_data.cat.categories:
                    new_cats = list(col_data.cat.categories) + [val]
                    new_dtype = pd.CategoricalDtype(
                        categories=new_cats, ordered=col_data.cat.ordered
                    )
                    df_filt[col] = df_filt[col].astype(new_dtype)
                df_filt.loc[:, col] = df_filt[col].fillna(val)
            else:
                df_filt.loc[:, col] = col_data.fillna(val)
    return df_filt


def prepare_shared_locations(shared: gpd.GeoDataFrame, params: dict) -> pd.DataFrame:
    """Format shared truck-stop charger locations with a standardised location-type label.

    Args:
        shared: Raw truck-stop GeoDataFrame (e.g., Jason's Law locations
            already assigned H3 hex IDs).
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names.
            - ``loc_col`` (str): output column name for the location type.
            - ``shared_location_type`` (str): categorical value assigned to
              all rows (e.g., ``"truck_stop"``).

    Returns:
        A ``pd.DataFrame`` with a categorical ``loc_col`` column.
    """
    shared = shared.rename(columns={v: k for k, v in params["col_renamer"].items()})
    shared[params["loc_col"]] = params["shared_location_type"]
    shared[params["loc_col"]] = pd.Categorical(shared[params["loc_col"]])
    return shared


def format_estabs(
    estabs_core: dd.DataFrame,
    estabs_geo: dd.DataFrame,
    estabs_rels: dd.DataFrame,
    params: dict,
) -> dgpd.GeoDataFrame:
    """Merge Data Axle establishment core, geo, and relationship tables into a single GeoDataFrame.

    The three raw Data Axle tables are joined on the establishment ID:
    ``estabs_core`` contains NAICS codes and employment; ``estabs_geo``
    contains latitude/longitude; ``estabs_rels`` contains parent-company
    relationships.  Establishments missing geographic coordinates are dropped.
    H3 resolution-8 hex IDs are computed from the coordinates.

    Args:
        estabs_core: Dask DataFrame of establishment business attributes.
        estabs_geo: Dask DataFrame of establishment geographic coordinates.
        estabs_rels: Dask DataFrame of parent-company relationship records.
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names, applied to all three tables.
            - ``default_naics`` (int): fallback NAICS code for establishments
              with a missing code.
            - ``calculated_keep_cols`` (list[str]): computed columns to retain
              (e.g., ``"hex_id"``, ``"geometry"``).

    Returns:
        A Dask GeoDataFrame with one row per establishment, point geometry,
        H3 hex-ID column, and the columns specified by ``col_renamer`` plus
        ``calculated_keep_cols``.
    """
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


def reassign_hqs(estabs: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Reassign corporate-HQ establishments to a HQ-specific NAICS code to prevent misclassification.

    A corporate headquarters may be coded with its parent company's primary
    NAICS (e.g., a trucking-company HQ coded as a freight terminal), which
    would cause the clustering step to treat the HQ building as a large freight
    facility.  This function identifies likely HQ records using a combination
    of employment ratio, business-status code, number of sibling establishments,
    and NAICS window, then overrides their NAICS code with ``params["hq_naics"]``.

    The heuristic is intentionally conservative (large employee ratio, specific
    HQ business codes, many sibling establishments) to avoid reassigning
    genuine large freight terminals.

    Args:
        estabs: Establishment GeoDataFrame with NAICS, employment, business-
            status, and parent-ID columns.
        params: Pipeline parameters dict with keys:

            - ``columns`` (dict): sub-keys ``parent_id``, ``naics``,
              ``n_employees``, ``buss_status``, ``estab_id``.
            - ``emp_ratio_min`` (float): minimum ``(establishment_employees /
              median_sibling_employees)`` ratio for HQ classification.
            - ``hq_bus_codes`` (list): business-status code values indicating
              an HQ.
            - ``n_estabs_big`` (int): minimum number of sibling establishments
              required.
            - ``naics_window`` (dict): ``lower`` and ``upper`` NAICS code
              bounds for the window within which HQ reassignment is applied.
            - ``hq_naics`` (int): the NAICS code to assign to identified HQ
              establishments.

    Returns:
        The establishment GeoDataFrame with HQ NAICS codes overridden.
    """
    pcols = params["columns"]
    parents = estabs.groupby(pcols["parent_id"]).agg(
        n_estabs_in_parent=pd.NamedAgg(pcols["naics"], "count"),
        n_emps_in_parent=pd.NamedAgg(pcols["n_employees"], "sum"),
        med_emps_in_parent=pd.NamedAgg(pcols["n_employees"], "median"),
    )

    estab_context = estabs.reset_index()
    estab_context = estab_context.merge(parents, how="left", on=pcols["parent_id"])
    estab_context = estab_context.set_index(pcols["estab_id"])

    estab_context["emp_ratio_to_med"] = (
        estab_context[pcols["n_employees"]] / estab_context["med_emps_in_parent"]
    )

    hqs = estab_context.loc[
        (estab_context["emp_ratio_to_med"] > params["emp_ratio_min"])
        & (estab_context[pcols["buss_status"]].isin(params["hq_bus_codes"]))
        & (estab_context["n_estabs_in_parent"] > params["n_estabs_big"])
        & (estab_context[pcols["naics"]] >= params["naics_window"]["lower"])
        & (estab_context[pcols["naics"]] < params["naics_window"]["upper"])
    ]

    estabs[pcols["naics"]] = estabs[pcols["naics"]].where(
        ~estabs.index.isin(hqs.index), params["hq_naics"]
    )

    return estabs


def prepare_stop_locations_public(
    parks: gpd.GeoDataFrame, params: dict
) -> gpd.GeoDataFrame:
    """Format Jason's Law truck-stop locations as establishment records.

    Assigns the truck-stop NAICS code and renames columns so that these
    public data records can be concatenated with the Data Axle establishment
    dataset.

    Args:
        parks: Jason's Law truck-stop GeoDataFrame.
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names.
            - ``columns`` (dict): sub-key ``naics`` for the NAICS column name.
            - ``naics_code`` (int): NAICS code to assign to all records.
            - ``keep_cols`` (list[str]): columns to retain.

    Returns:
        A ``gpd.GeoDataFrame`` ready to be concatenated with ``format_estabs``
        output.
    """
    pcols = params["columns"]
    parks_fmt = parks.rename(columns={v: k for k, v in params["col_renamer"].items()})
    parks_fmt[pcols["naics"]] = params["naics_code"]
    parks_fmt = parks_fmt.loc[:, params["keep_cols"]]
    return parks_fmt


def get_osm_estabs_truck_stops(osm_params: dict, params: dict) -> gpd.GeoDataFrame:
    """Extract truck-stop fuel stations from an OSM PBF file by name pattern.

    Filters OSM nodes/ways that have a ``name`` tag, are tagged as
    ``amenity=fuel``, and whose name matches ``params["tag_regex"]`` (e.g.,
    ``"(?i)truck stop|travel center"``).

    Args:
        osm_params: OSM server/path configuration dict with keys:

            - ``osm_path`` (str): path to the OSM PBF input file.
            - ``temp_path`` (str): path for temporary files during OSM parsing.
        params: Pipeline parameters dict with keys:

            - ``naics_code`` (int): NAICS code to assign to all extracted
              records.
            - ``tag_regex`` (str): regular-expression pattern matched against
              the OSM ``name`` tag.
            - ``naics_col`` (str): output column name for the NAICS code.

    Returns:
        A ``gpd.GeoDataFrame`` of matching OSM establishments with a
        ``naics_col`` column added.
    """
    naics_code = params["naics_code"]
    filts = [
        KeyFilter("name"),
        TagFilter(("amenity", "fuel")),
        RegexTagFilter(tag="name", pattern=params["tag_regex"]),
    ]

    gdf = get_gdf_from_filtered_osm(
        osm_path=osm_params["osm_path"],
        filters=filts,
        tags=["name"],
        temp_path=osm_params["temp_path"],
    )
    gdf[params["naics_col"]] = naics_code
    return gdf


def get_osm_estabs_warehouses(osm_params: dict, params: dict) -> gpd.GeoDataFrame:
    """Extract warehouse and distribution-centre candidates from an OSM PBF file by name pattern.

    Filters OSM nodes/ways that have a ``name`` tag and whose name matches
    ``params["tag_regex"]`` (e.g., ``"(?i)warehouse|distribution"``).
    Records tagged as ``amenity=social_facility`` are excluded as false
    positives.

    Args:
        osm_params: OSM server/path configuration dict with keys:

            - ``osm_path`` (str): path to the OSM PBF input file.
            - ``temp_path`` (str): path for temporary files during OSM parsing.
        params: Pipeline parameters dict with keys:

            - ``naics_code`` (int): NAICS code to assign to all extracted
              records.
            - ``tag_regex`` (str): regular-expression pattern matched against
              the OSM ``name`` tag.
            - ``naics_col`` (str): output column name for the NAICS code.

    Returns:
        A ``gpd.GeoDataFrame`` of matching OSM establishments (excluding
        social facilities) with a ``naics_col`` column added.
    """
    naics_code = params["naics_code"]
    filts = [
        KeyFilter("name"),
        RegexTagFilter(tag="name", pattern=params["tag_regex"]),
    ]

    gdf = get_gdf_from_filtered_osm(
        osm_path=osm_params["osm_path"],
        filters=filts,
        tags=["name", "amenity"],
        temp_path=osm_params["temp_path"],
    )
    gdf[params["naics_col"]] = naics_code

    gdf_filt = gdf.loc[gdf["amenity"] != "social_facility"]
    gdf_filt = gdf_filt.drop(columns=["amenity"])
    return gdf_filt


def concat_extra_estabs(*args: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Concatenate supplementary establishment GeoDataFrames from multiple sources.

    Args:
        *args: Two or more ``gpd.GeoDataFrame`` objects sharing the same
            column schema.

    Returns:
        A single ``gpd.GeoDataFrame`` with all rows concatenated and a fresh
        integer index.
    """
    estabs_concat = pd.concat(args, ignore_index=True, axis=0)
    return estabs_concat


def format_extra_estabs(estabs: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Reproject, compute H3 hex IDs, and deduplicate supplementary establishment records.

    Converts polygon geometries to centroids in a projected CRS, reprojects
    to the H3 geographic CRS, computes H3 resolution-8 hex IDs, and drops
    duplicate (hex, NAICS, name) combinations.

    Args:
        estabs: Concatenated supplementary establishment GeoDataFrame (output
            of ``concat_extra_estabs``).
        params: Pipeline parameters dict with keys:

            - ``proj_crs`` (str | CRS): projected CRS for centroid computation.
            - ``hex_col`` (str): output H3 hex-ID column name.
            - ``naics_col`` (str): NAICS column name (used for deduplication).
            - ``name_col`` (str): name column name (used for deduplication).

    Returns:
        A ``gpd.GeoDataFrame`` of establishments with ``uint64`` H3 hex IDs
        and duplicates removed.
    """
    estabs_centers = estabs.to_crs(params["proj_crs"])
    estabs_centers[estabs_centers.geometry.name] = estabs_centers.geometry.centroid
    estabs_centers = estabs_centers.to_crs(H3_CRS)
    estabs_centers[params["hex_col"]] = coords_to_cells(
        lat=estabs_centers.geometry.y.values,
        lng=estabs_centers.geometry.x.values,
        res=H3_DEFAULT_RESOLUTION,
    )
    dup_subset = [params["hex_col"], params["naics_col"], params["name_col"]]
    estabs_centers = estabs_centers.loc[~estabs_centers.duplicated(subset=dup_subset)]
    estabs_centers[params["hex_col"]] = estabs_centers[params["hex_col"]].astype(
        np.uint64
    )
    return estabs_centers


def collapse_naics_classes(
    estabs: gpd.GeoDataFrame, naics_leaves: pd.DataFrame, params: dict
) -> gpd.GeoDataFrame:
    """Map 8-digit NAICS codes to the smaller set of leaf classes used for clustering.

    Each raw NAICS code is mapped to the most specific leaf in ``naics_leaves``
    that is a prefix of the raw code.  Codes with no matching leaf are assigned
    ``params["fill_leaf"]``.  This reduces the dimensionality of the employment
    embedding while preserving the distinctions most relevant to freight
    activity.

    Args:
        estabs: Establishment GeoDataFrame with an 8-digit NAICS column.
        naics_leaves: DataFrame of allowed leaf NAICS codes.
        params: Pipeline parameters dict with keys:

            - ``naics_cols`` (dict): sub-keys ``raw`` (input column) and
              ``out`` (output column) and ``leaf`` (column in ``naics_leaves``
              containing the allowed codes).
            - ``fill_leaf`` (int): leaf code to use when no match is found.

    Returns:
        The establishment GeoDataFrame with ``params["naics_cols"]["out"]``
        added as an integer column.
    """
    ncols = params["naics_cols"]
    estabs[ncols["out"]] = get_naics_leaf_class(
        codes=estabs[ncols["raw"]].values,
        leaves=naics_leaves[ncols["leaf"]].values,
        fill_leaf=params["fill_leaf"],
    ).astype(int)
    return estabs


def pivot_hex_estabs(estabs: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Build a per-hexagon employment embedding matrix from the establishment dataset.

    For each hexagon, computes the total employment in each NAICS leaf class
    across all establishments in that hexagon.  Establishments reporting zero
    employees are filled with the median employment for that NAICS class, then
    offset by ``params["zero_emp_buff"]`` to avoid log-zero issues downstream.

    Column names for NAICS codes are prefixed with ``params["naics_prefix"]``
    and sorted lexically to ensure stable column ordering across pipeline runs.

    Args:
        estabs: Establishment GeoDataFrame with leaf NAICS codes and employment
            (output of ``collapse_naics_classes``).
        params: Pipeline parameters dict with keys:

            - ``columns`` (dict): sub-keys ``hex_id``, ``naics``,
              ``n_employees``, ``geom``.
            - ``naics_prefix`` (str): string prefix for NAICS column names
              (e.g., ``"naics_"``).
            - ``zero_emp_buff`` (float): small constant added to all employee
              counts after median-fill to prevent exact zeros.
            - ``keep_metadata_cols`` (list[str]): non-NAICS attribute columns
              to retain in the output (e.g., substation territory ID).

    Returns:
        A ``gpd.GeoDataFrame`` indexed by hexagon ID with one column per
        NAICS leaf class (integer employment totals) plus metadata columns.
    """
    pcols = params["columns"]
    emp_col = pcols["n_employees"]

    med_emp_by_naics = (
        estabs.loc[estabs[emp_col] > 0].groupby(pcols["naics"])[emp_col].median()
    )
    fill_emps = (estabs[emp_col] == 0) | estabs[emp_col].isna()
    estabs[emp_col] = estabs[emp_col].where(
        ~fill_emps, estabs[pcols["naics"]].map(med_emp_by_naics)
    )
    estabs[emp_col] = estabs[emp_col].fillna(0)
    estabs[emp_col] = estabs[emp_col] + (estabs[emp_col] == 0) * params["zero_emp_buff"]

    logger.info("Computing embedding for each hexagon")
    estabs_hex = estabs.groupby([pcols["hex_id"], pcols["naics"]])[emp_col].sum()
    estabs_hex = estabs_hex.unstack(pcols["naics"], fill_value=0).astype(int)

    # Establishing lexical sorting of NAICS code strings
    pre = params["naics_prefix"]
    naics_to_str = {
        col: f"{pre}{col}" for col in estabs_hex.columns if isinstance(col, int)
    }
    estabs_hex = estabs_hex.rename(columns=naics_to_str)
    estabs_hex = estabs_hex.sort_index(axis=1)

    mrg_cols = [pcols["hex_id"], pcols["geom"]] + params["keep_metadata_cols"]
    hex_mrg = estabs.loc[:, mrg_cols].drop_duplicates(pcols["hex_id"])
    hex_mrg = hex_mrg.set_index(pcols["hex_id"])
    estabs_hex = estabs_hex.merge(
        hex_mrg, how="left", left_index=True, right_index=True
    )
    estabs_hex = gpd.GeoDataFrame(data=estabs_hex, geometry=pcols["geom"])
    return estabs_hex


def partition_hex_corresp(
    hex_base: pd.DataFrame, params: dict, client: Client | str
) -> dd.DataFrame:
    """Split the hexagon base correspondence table into Dask parquet partitions.

    Converts the pandas feather ``hex_base_corresp`` to a Dask DataFrame so that
    the downstream :func:`read_land_use` node can extract raster values in parallel,
    one partition per worker.  The index is sorted before partitioning so that
    each partition covers a contiguous range of uint64 hex IDs.

    Args:
        hex_base: Pandas DataFrame of hexagon base correspondence data, indexed
            by uint64 hex ID.
        params: Pipeline parameters dict with key:

            - ``n_partitions`` (int): number of Dask partitions to create.
              Controls parallelism during raster extraction.
        client: Active Dask ``Client`` (or the ``"None"`` sentinel when Dask is
            disabled).  Not used directly; accepted here solely to enforce DAG
            ordering so this node does not execute until the Dask cluster is up.

    Returns:
        Dask DataFrame with the same schema as ``hex_base``, split into
        ``n_partitions`` partitions and saved as ``hex_base_corresp_dask``.
    """
    hex_base = hex_base.sort_index()
    return dd.from_pandas(hex_base, npartitions=params["n_partitions"])


def _extract_land_use_part(
    part: gpd.GeoDataFrame, raster_path: Path, idx_col: str, **kwargs
) -> pd.DataFrame:
    """Extract NLCD land cover fractions for one Dask partition of hex polygons.

    Wraps :func:`exactextract.exact_extract` to compute, for each hexagon polygon
    in ``part``, the unique NLCD category codes present and their fractional coverage.
    Returns a long-format DataFrame with one row per (hexagon, category) pair.

    Args:
        part: GeoDataFrame partition of hexagon polygons.  Must contain a string
            column named ``idx_col`` that uniquely identifies each hexagon.
        raster_path: Filesystem path to the NLCD 2023 GeoTIFF.  Passed directly
            to ``exactextract``; file I/O is handled by the library.
        idx_col: Name of the string hexagon ID column to use as the output index.
        **kwargs: Additional keyword arguments forwarded to
            :func:`exactextract.exact_extract`.

    Returns:
        DataFrame indexed by ``idx_col`` with columns:

        - ``unique`` (int32): NLCD land cover category code.
        - ``frac`` (float64): fraction of hexagon area covered by that category.
    """
    ops = ["unique", "frac"]
    extracted: pd.DataFrame = ex.exact_extract(
        rast=raster_path,
        vec=part,
        include_cols=idx_col,
        output="pandas",
        ops=ops,
        **kwargs,
    )
    extracted = extracted.set_index(idx_col)
    extracted = extracted.explode(column=ops)
    return extracted


def read_land_use(hex_base_corresp: dd.DataFrame, params: dict) -> dd.DataFrame:
    """Extract NLCD 2023 land cover fractions for every H3 hexagon using exactextract.

    Reads the NLCD land cover raster directly via ``exactextract``, computing for
    each hexagon polygon the fraction of its area covered by each NLCD category code.
    The result is in long format: one row per (hexagon × category) pair.

    The algorithm proceeds in three steps:

    1. Attach polygon geometries to each partition of ``hex_base_corresp`` using
       :func:`~laurel.utils.h3.add_geometries`, producing a Dask GeoDataFrame.
    2. Reproject each partition to the raster's native CRS (read once from the
       file header) so that fractional areas are computed in the correct coordinate
       space.
    3. Call :func:`_extract_land_use_part` via ``map_partitions``, passing the
       raster path directly so ``exactextract`` handles file I/O per partition.

    .. note::
        This node is I/O-intensive and may take several hours to complete on a
        single machine.  Run it once with
        ``kedro run --pipeline=describe_locations --tags=read_land_use`` and rely
        on the cached ``hex_land_use`` parquet output for subsequent runs.

    The index of the output is a string representation of the uint64 hex ID
    (``{hex_col}_str``).  The downstream :func:`pivot_hex_land_use` node casts
    it back to ``uint64`` and renames it to ``{hex_col}``.

    Args:
        hex_base_corresp: Dask DataFrame of hexagon correspondence data, indexed
            by uint64 hex ID.  Loaded from ``hex_base_corresp_dask`` in the catalog.
        params: Pipeline parameters dict with keys:

            - ``raster_path`` (str): path to the NLCD 2023 GeoTIFF, relative to
              the Kedro project root.  Passed directly to ``exactextract`` rather
              than loaded through the Kedro catalog because there is no standard
              dataset type that reads a GeoTIFF into the format ``exactextract``
              requires.
            - ``hex_col`` (str): name of the hex ID column / index; used to derive
              the string column ``{hex_col}_str`` that becomes the output index.

    Returns:
        Dask DataFrame indexed by ``{hex_col}_str`` (object dtype) with columns:

        - ``unique`` (int32): NLCD land cover category code.
        - ``frac`` (float64): fraction of hexagon area covered by that category.
    """
    raster_path = Path(params["raster_path"])
    hex_col = params["hex_col"]
    idx_col = f"{hex_col}_str"

    hex_base = hex_base_corresp.assign(**{hex_col: lambda ddf: ddf.index})
    hex_base_geo = add_geometries(hex_base, hex_col=hex_col, geom_type="polygon")

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    hex_base_proj = hex_base_geo.to_crs(raster_crs)
    hex_base_proj[idx_col] = hex_base_proj[hex_col].astype(str)

    meta_df = pd.DataFrame(
        {"unique": pd.Series(dtype="int32"), "frac": pd.Series(dtype="float64")}
    ).set_index(pd.Index([], name=idx_col, dtype="object"))

    hex_extracted = hex_base_proj.map_partitions(
        _extract_land_use_part,
        raster_path=raster_path,
        idx_col=idx_col,
        meta=meta_df,
    )

    return hex_extracted


def pivot_hex_land_use(land_use: dd.DataFrame, params: dict) -> pd.DataFrame:
    """Pivot the per-hexagon NLCD land-use coverage table from long to wide format.

    Maps fine-grained NLCD category codes to broader land-use groups using
    ``params["code_group_corresp"]``, then aggregates fractional coverage by
    (hexagon, group) and unstacks to produce one column per land-use group.

    The hexagon index is cast to ``uint64`` to match the dtype used elsewhere
    in the pipeline.

    Args:
        land_use: Dask DataFrame of NLCD coverage fractions in long format,
            indexed by hexagon ID string.
        params: Pipeline parameters dict with keys:

            - ``input_cols`` (dict): sub-keys ``categories`` (NLCD code
              column), ``fractions`` (coverage fraction column), ``hex``
              (hexagon index name after renaming).
            - ``code_group_corresp`` (dict[str, str]): mapping from NLCD code
              to land-use group label.

    Returns:
        A ``pd.DataFrame`` indexed by hexagon ID (``uint64``) with one column
        per land-use group, filled with 0.0 for groups with no coverage.
    """
    ccol = params["input_cols"]["categories"]
    fcol = params["input_cols"]["fractions"]
    hcol = params["input_cols"]["hex"]
    land_use["land_use_group"] = land_use[ccol].map(
        params["code_group_corresp"], meta=(ccol, "str")
    )
    land_use["land_use_group"] = land_use["land_use_group"].astype("category")
    land_use = land_use.drop(columns=[ccol])
    meta_uint64 = land_use._meta.copy()
    meta_uint64.index = meta_uint64.index.astype(np.uint64)
    land_use = land_use.map_partitions(
        lambda part: part.set_axis(part.index.astype(np.uint64), axis=0),
        meta=meta_uint64,
    )

    land_use = land_use.rename_axis(index={f"{hcol}_str": hcol})

    def groupby_part(part: pd.DataFrame) -> pd.DataFrame:
        return part.groupby([hcol, "land_use_group"], observed=True)[fcol].sum()

    land_use = land_use.map_partitions(groupby_part)
    land_use_hex = land_use.compute()
    land_use_hex = land_use_hex.unstack("land_use_group", fill_value=0.0)
    return land_use_hex


def group_hexes(
    land_use: pd.DataFrame, estabs: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Assign each hexagon a freight-activity class using development thresholds, rules, and K-Means.

    The assignment algorithm proceeds in five stages:

    1. **Development threshold**: Only hexagons with at least
       ``params["development"]["frac_thresh"]`` developed land cover are
       considered "developed" and are eligible for freight-class assignment.
       All others receive the label ``"undeveloped"``.
    2. **Neighbor embedding**: Each developed hexagon's employment vector is
       augmented with the summed employment of its ring-1 H3 neighbors
       (``include_center=False, distance=1``), capturing the surrounding
       land-use context.
    3. **Special establishment detection**: Hexagons containing (or adjacent
       to) any establishment with a NAICS code listed in
       ``params["special_naics"]`` are labelled with that special class
       (e.g., ``"truck_stop"``).  Special labels take precedence over
       cluster-assigned labels.
    4. **K-Means clustering**: The remaining developed hexagons with at least
       one freight-intensive NAICS establishment are clustered using K-Means
       on the log1p-transformed sparse employment matrix.  Hexagons with no
       freight-intensive establishments receive the label ``"some_estabs"``
       or ``"no_estabs"``.
    5. **Output**: All labels (including ``"undeveloped"``) are consolidated
       into a single categorical column.

    Args:
        land_use: Per-hexagon land-use-group fraction matrix (output of
            ``pivot_hex_land_use``).
        estabs: Per-hexagon employment embedding matrix (output of
            ``pivot_hex_estabs``).
        params: Pipeline parameters dict with keys:

            - ``development`` (dict): ``col`` (developed land-cover column)
              and ``frac_thresh`` (minimum fraction threshold).
            - ``naics_prefix`` (str): prefix identifying NAICS columns.
            - ``default_naics`` (int): NAICS code used as the "no freight
              industry" category (excluded from freight-intensity tests).
            - ``special_naics`` (dict[str, int]): mapping from label to
              NAICS code for special establishment categories.
            - ``loc_group_col`` (str): output freight-activity-class column
              name.
            - ``clusterer_kwargs`` (dict): keyword arguments forwarded to
              ``sklearn.cluster.KMeans``.

    Returns:
        A ``pd.DataFrame`` indexed by hexagon ID with a single categorical
        column ``params["loc_group_col"]`` containing the freight-activity
        class for each hexagon.
    """
    # Mark developed hexes
    dpars = params["development"]
    land_use["is_developed"] = land_use[dpars["col"]] >= dpars["frac_thresh"]

    # Select only developed hexes and fill in their estabilishments
    hex_dev = land_use.loc[
        land_use["is_developed"]
    ]  # Applies known pattern of only trusting that establishments can only exist where there is a minimum level of development
    hex_dev = hex_dev.join(estabs)
    naics_cols = [
        col for col in estabs.columns if col.startswith(params["naics_prefix"])
    ]
    for ncol in naics_cols:
        hex_dev[ncol] = hex_dev[ncol].fillna(0)

    # Get neighbor establishments for developed hexes
    nbr_embs = get_neighbor_embeddings(
        hexes=hex_dev.index.values.astype(np.uint64),
        embs=hex_dev.loc[:, naics_cols].values,
        include_center=False,
        distance=1,
    )
    nbr_df = pd.DataFrame(data=nbr_embs, columns=naics_cols, index=hex_dev.index)

    def get_nbr_cols(cols: list[str]) -> list[str]:
        return [f"{col}_nbr" for col in cols]

    renamer = {orig: new for orig, new in zip(naics_cols, get_nbr_cols(naics_cols))}
    mrg = nbr_df.rename(columns=renamer)
    hex_dev = pd.concat([hex_dev, mrg], axis=1)

    # Mark hexes with any establishments at all
    all_fi_cols = naics_cols + get_nbr_cols(naics_cols)
    hex_dev["has_any_estabs"] = hex_dev.loc[:, all_fi_cols].sum(axis=1) > 0

    # Mark hexes with any freight-intensive establishments
    default_code = params["naics_prefix"] + str(params["default_naics"])
    naics_fi_cols = list(set(naics_cols).difference([default_code]))
    all_fi_cols = naics_fi_cols + get_nbr_cols(naics_fi_cols)
    hex_dev["has_any_fi"] = hex_dev.loc[:, all_fi_cols].sum(axis=1) > 0

    # Mark hexes with special establishments
    hex_dev["has_any_special"] = False
    for name, code in params["special_naics"].items():
        str_code = params["naics_prefix"] + str(code)
        spec_cols = [str_code] + get_nbr_cols([str_code])
        hex_dev["has_any_special"] |= hex_dev.loc[:, spec_cols].sum(axis=1) > 0

    # Cluster non-special freight-intensive hexes
    gcol = params["loc_group_col"]
    feature_cols = naics_cols + get_nbr_cols(naics_cols)
    features = hex_dev.loc[
        hex_dev["has_any_fi"] & ~hex_dev["has_any_special"], feature_cols
    ]

    feature_sparse = features.astype(pd.SparseDtype("float", 0.0))
    X_train = feature_sparse.sparse.to_coo()
    X_transf = X_train.log1p()
    clusterer = KMeans(**params["clusterer_kwargs"])
    clusterer = clusterer.fit(X=X_transf)
    clusts = pd.Series(clusterer.labels_, index=features.index, name=gcol)

    clust_prefix = "clust_"
    clusts = clust_prefix + clusts.astype(str)

    # Collate categories for developed hexes
    hex_dev[gcol] = "no_estabs"
    hex_dev[gcol] = hex_dev[gcol].where(~hex_dev["has_any_estabs"], "some_estabs")
    # TODO: Mapping clusts directly onto land_use produces better homogeneity. Why?
    hex_dev[gcol] = hex_dev[gcol].where(
        ~hex_dev["has_any_fi"], hex_dev.index.map(clusts)
    )

    for name, code in params["special_naics"].items():
        str_code = params["naics_prefix"] + str(code)
        spec_cols = [str_code]
        any_special = hex_dev.loc[:, spec_cols].sum(axis=1) > 0
        hex_dev[gcol] = hex_dev[gcol].where(~any_special, name)

    # Collate categories for all hexes
    land_use[gcol] = land_use.index.map(hex_dev[gcol])
    land_use[gcol] = land_use[gcol].fillna("undeveloped")
    land_use[gcol] = land_use[gcol].astype("category")

    return land_use.loc[:, [gcol]]


def apply_groups(
    hexes: pd.DataFrame, groups: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Merge freight-activity-class labels onto a hexagon feature table.

    Args:
        hexes: Per-hexagon feature DataFrame indexed by hex ID.
        groups: Freight-activity-class labels indexed by hex ID (output of
            ``group_hexes``).
        params: Pipeline parameters dict with keys:

            - ``hex_col`` (str): name of the shared hexagon index.

    Returns:
        The ``hexes`` DataFrame with the freight-activity-class column
        from ``groups`` merged in on the shared hex index.

    Raises:
        AssertionError: If either ``hexes`` or ``groups`` does not have
            ``params["hex_col"]`` as its index name.
    """
    assert hexes.index.name == params["hex_col"]
    assert groups.index.name == params["hex_col"]
    out = hexes.merge(groups, how="left", left_index=True, right_index=True)
    return out
