"""
This is a boilerplate pipeline 'describe_locations'
generated using Kedro 0.19.3
"""

import logging

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.dataframe.dispatch import make_meta
from dask.diagnostics.progress import ProgressBar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from megaPLuG.utils.h3 import (
    H3_CRS,
    H3_DEFAULT_RESOLUTION,
    coords_to_cells_wrapper,
    region_polygons_to_cells,
)
from megaPLuG.utils.hex_neighbors import get_neighbor_embeddings
from megaPLuG.utils.naics import get_naics_leaf_class

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
    subs[params["add_source_col"]["name"]] = params["add_source_col"]["value"]
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


def format_substations_contin(subs: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Get substation polygons from points and state boundaries."""
    subs = subs.rename(columns={v: k for k, v in params["col_renamer"].items()})
    subs["substation_id"] = subs["substation_id"].astype(int)
    subs[params["add_source_col"]["name"]] = params["add_source_col"]["value"]
    subs = subs.loc[:, params["keep_cols"]]
    return subs


def fill_out_substations(
    poly_subs: gpd.GeoDataFrame, point_subs: gpd.GeoDataFrame, params: dict
) -> gpd.GeoDataFrame:
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
    """Build substation territories to cover the whole analysis area using point-based
    substation data to fill in the remainder of what is left after polygon-based substation
    data is used.
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
    """Format the states dataset."""
    states = states.rename(columns={v: k for k, v in params["col_renamer"].items()})
    return states


def format_urban(urban: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Format the urban areas dataset."""
    urban = urban.rename(columns={v: k for k, v in params["col_renamer"].items()})
    return urban


def format_highways(highways: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Format the highways dataset."""
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


def clip_to_extent(
    gdf: gpd.GeoDataFrame, extent: gpd.GeoDataFrame, params: dict
) -> pd.DataFrame:
    """Clip a geometry layer to an extent layer."""
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
    """Hexify the polygons in a GeoDataFrame."""

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
    """Concatenate columns by their shared index."""
    cat = pd.concat(args, axis=1, join="outer")
    return cat


def fill_missingness(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Fill in missingness using the parameters."""
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
    """Prepare the charging locations shared by all vehicles."""
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


def reassign_hqs(estabs: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Re-assign headquarters establishments to a headquarters NAICS code.

    This allows us to avoid seeing a headquarters for a truck stop company, for instance,
    as the world's biggest truck stop.
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


def collapse_naics_classes(
    estabs: gpd.GeoDataFrame, naics_leaves: pd.DataFrame, params: dict
) -> gpd.GeoDataFrame:
    """Collapse the NAICS classes down to the specific level of detail we need."""
    ncols = params["naics_cols"]
    estabs[ncols["out"]] = get_naics_leaf_class(
        codes=estabs[ncols["raw"]].values,
        leaves=naics_leaves[ncols["leaf"]].values,
    ).astype(int)
    return estabs


def embed_hexes(estabs: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Create embeddings for each hexagon based on the total employment in each establishment type."""
    pcols = params["columns"]
    emp_col = pcols["n_employees"]
    estabs[emp_col] = estabs[emp_col] + (estabs[emp_col] == 0) * params["zero_emp_buff"]

    logger.info("Computing embedding for each hexagon")
    estabs_hex = estabs.groupby([pcols["hex_id"], pcols["naics"]])[emp_col].sum()
    estabs_hex = estabs_hex.unstack(pcols["naics"], fill_value=0).astype(int)

    pre = params["naics_prefix"]
    naics_to_str = {
        col: f"{pre}{col}" for col in estabs_hex.columns if isinstance(col, int)
    }
    estabs_hex = estabs_hex.rename(columns=naics_to_str)
    estabs_hex = estabs_hex.sort_index(
        axis=1
    )  # Establishing lexical sorting of NAICS code strings

    mrg_cols = [pcols["hex_id"], pcols["geom"]] + params["keep_metadata_cols"]
    hex_mrg = estabs.loc[:, mrg_cols].drop_duplicates(pcols["hex_id"])
    hex_mrg = hex_mrg.set_index(pcols["hex_id"])
    estabs_hex = estabs_hex.merge(
        hex_mrg, how="left", left_index=True, right_index=True
    )
    estabs_hex = gpd.GeoDataFrame(data=estabs_hex, geometry=pcols["geom"])

    if params["include_neighbors"]:
        logger.info("Computing neighbor embeddings for each hexagon")
        naics_cols = sorted(list(naics_to_str.values()))
        ngbr_embs = get_neighbor_embeddings(
            hexes=estabs_hex.index.values.astype(np.uint64),
            embs=estabs_hex.loc[:, naics_cols].values,
            include_center=False,
            distance=1,
        )
        ngbr_df = pd.DataFrame(
            data=ngbr_embs, columns=naics_cols, index=estabs_hex.index
        )
        estabs_hex = estabs_hex.merge(
            ngbr_df,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=("", params["neighbor_merge_suffix"]),
        )

    return estabs_hex


def cluster_hexes(embs: gpd.GeoDataFrame, params: dict) -> pd.DataFrame:
    """Cluster hexagons together based on embeddings."""
    scaler = StandardScaler(**params["scaler_kwargs"])
    clusterer = KMeans(**params["clusterer_kwargs"])

    fcols = []
    if params["features"]["include_own_naics"]:
        fcols.extend(
            [col for col in embs.columns if col.startswith(params["naics_prefix"])]
        )
    if params["features"]["include_ngbr_naics"]:
        fcols.extend(
            [col for col in embs.columns if col.endswith(params["ngbr_suffix"])]
        )

    other_feats = params["features"]["other_features"]
    if other_feats is not None:
        fcols.extend(other_feats)

    X_train = embs.loc[:, fcols].values
    X_transf = np.log10(1 + X_train)
    train_scaled = scaler.fit_transform(X=X_transf)
    embs[params["cluster_col"]] = clusterer.fit_predict(X=train_scaled)

    return embs.loc[:, [params["cluster_col"]]]


def apply_clusters(
    hexes: pd.DataFrame, clusts: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Apply clusters to locations of observed dwells."""
    assert hexes.index.name == params["hex_col"]
    assert clusts.index.name == params["hex_col"]
    out = hexes.merge(clusts, how="left", left_index=True, right_index=True)
    cl_col = params["clust_col"]
    if params["fill_cluster"]:
        out[cl_col] = out[cl_col].fillna(params["fill_cluster_value"]).astype(int)
    return out
