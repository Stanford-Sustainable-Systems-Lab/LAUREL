import geopandas as gpd


def find_time_weighted_centers(
    gdf: gpd.GeoDataFrame, grp_col: str, weight_col: str
) -> gpd.GeoDataFrame:
    """Find time-weighted center of a set of dwells.

    WARNING: This function may be very slow in Dask if the gdf is not indexed on grp_col
    """
    if not hasattr(gdf, "crs"):
        raise RuntimeError("The DwellSet's underlying dataset is not geographic.")
    else:
        crs = gdf.crs
        if not crs.is_projected:
            raise RuntimeError(
                "The DwellSet's underlying dataset is not in projected coordinates."
            )

    gdf["easting_wt"] = gdf.geometry.x * gdf[weight_col]
    gdf["northing_wt"] = gdf.geometry.y * gdf[weight_col]
    centers = gdf.groupby(grp_col).agg(
        {"easting_wt": "sum", "northing_wt": "sum", weight_col: "sum"}
    )
    gdf = gdf.drop(columns=["easting_wt", "northing_wt"])
    centers["easting"] = centers["easting_wt"] / centers[weight_col]
    centers["northing"] = centers["northing_wt"] / centers[weight_col]
    geoms = gpd.GeoSeries.from_xy(x=centers["easting"], y=centers["northing"], crs=crs)
    centers = gpd.GeoDataFrame(index=centers.index, geometry=geoms.values)
    return centers
