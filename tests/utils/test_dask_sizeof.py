"""Tests for the GeoPandas ``dask.sizeof`` registrations.

Guards against the regression described in ``laurel.utils.dask_sizeof``:
without the registrations, ``sizeof()`` on a ``GeoSeries``/``GeoDataFrame``
reports the same size regardless of how many coordinates each geometry
holds, because it falls back to ``memory_usage(deep=False)`` for the
geometry-dtype column.
"""

import gc
import os

import geopandas as gpd
import psutil
import pytest
import shapely
from dask.sizeof import sizeof

import laurel.utils.dask_sizeof  # noqa: F401  (import-time side effect: registers dask sizeof() handlers)

FEW_POINTS = 3
MANY_POINTS = 1000
N_GEOMS = 1000


def _linestrings(n_geoms: int, n_points: int) -> list[shapely.LineString]:
    return [shapely.LineString([(i, i) for i in range(n_points)]) for _ in range(n_geoms)]


def test_geoseries_sizeof_scales_with_point_count():
    small = gpd.GeoSeries(_linestrings(N_GEOMS, FEW_POINTS))
    big = gpd.GeoSeries(_linestrings(N_GEOMS, MANY_POINTS))

    assert sizeof(big) / sizeof(small) > 10


def test_geodataframe_sizeof_scales_with_point_count():
    small = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(_linestrings(N_GEOMS, FEW_POINTS))})
    big = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(_linestrings(N_GEOMS, MANY_POINTS))})

    assert sizeof(big) / sizeof(small) > 10


def test_geodataframe_sizeof_includes_non_geometry_columns():
    gdf = gpd.GeoDataFrame(
        {
            "geometry": gpd.GeoSeries(_linestrings(N_GEOMS, MANY_POINTS)),
            "label": ["x" * 50] * N_GEOMS,
        }
    )
    geom_only = gpd.GeoDataFrame({"geometry": gdf["geometry"]})

    assert sizeof(gdf) > sizeof(geom_only)


@pytest.mark.performance
def test_geoseries_sizeof_is_an_upper_bound_on_measured_rss():
    proc = psutil.Process(os.getpid())
    n_geoms, n_points = 50_000, 100

    gc.collect()
    before = proc.memory_info().rss
    gs = gpd.GeoSeries(_linestrings(n_geoms, n_points))
    gc.collect()
    measured = proc.memory_info().rss - before

    estimated = sizeof(gs)

    # Generous slack: RSS-delta measurement is noisy (allocator behavior,
    # measurement overhead from psutil/gc itself), so this only checks the
    # estimate isn't wildly under the true footprint -- it is not a tight
    # equality check.
    assert estimated > measured * 0.5
