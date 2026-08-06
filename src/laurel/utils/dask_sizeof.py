"""Accurate ``dask.sizeof`` registrations for GeoPandas geometry columns.

Dask's built-in ``sizeof()`` handlers for ``pd.Series``/``pd.DataFrame``
only fall back to a coordinate-aware measurement when a column's dtype is
in a small hard-coded set of "object-like" dtypes. ``geopandas.array.
GeometryDtype`` is not in that set, so Dask instead reports
``memory_usage(deep=False)`` for geometry columns — for an extension array
that is just ``n_rows * 8`` (the pointer slots), independent of how many
coordinates each geometry actually holds. A GeoSeries of LineStrings with
3 points each and one with 1,000 points each report identically wrong
sizes. Since ``sizeof()`` drives Dask's spill-to-disk memory management,
this silent under-count risks worker OOM before a spill is ever triggered.

This module registers replacement handlers for ``shapely.Geometry``,
``geopandas.array.GeometryArray``, ``geopandas.GeoSeries``, and
``geopandas.GeoDataFrame`` that estimate size from vertex counts using
``shapely.get_num_coordinates`` (a single vectorized C call over the whole
underlying object array — no per-geometry Python loop), so the estimate
stays cheap enough to call on every scheduling/spill decision.

The estimate is deliberately a padded upper bound, not a tight average:
Dask uses ``sizeof()`` to decide when workers are at risk of running out
of memory, so overestimating is safe (extra, avoidable spills) while
underestimating is not (OOM).

The constants below were calibrated against REAL project data (`parking.gpkg`
plus a random sample of 30 real `trips_routed` partitions), not synthetic
geometries. An earlier version of this module calibrated
``BYTES_PER_COORD``/``GEOM_OVERHEAD_BYTES`` by building uniform-size
``shapely.linestrings()`` batches and measuring RSS deltas — that
under-counted real data by 2-700x, for two independent reasons:

1. **Allocation pattern.** A single batch call allocates uniform-size
   buffers efficiently; real ``trips_routed`` partitions deserialize a wide
   mix of geometry sizes one at a time from individual WKB blobs during
   ``read_parquet``, which fragments and costs more per coordinate (measured
   50-107 bytes/coordinate on real partitions, isolated per-partition in a
   fresh subprocess to avoid allocator-reuse noise from prior
   measurements — vs. ~24-32 bytes/coordinate from synthetic batches).
2. **Missing regime.** Synthetic calibration only used vertex-heavy
   LineStrings, so the fitted ``GEOM_OVERHEAD_BYTES`` (fixed per-geometry
   cost, independent of vertex count) rounded to ~0 — fine for large
   LineStrings where per-coordinate cost dominates, but catastrophically
   wrong for point-heavy data like `parking.gpkg` (2783 Points, ~1
   coordinate each), where the real per-geometry cost is ~7000 bytes and
   coordinate count contributes almost nothing. A model tuned only on
   LineStrings underestimated that case by ~50x.

:func:`calibrate_from_real_data` fixes both: it measures each sample in an
isolated subprocess (a fresh process per file, so one measurement's
allocator state can't contaminate the next — the same kind of allocator
retention this module's ``sizeof()`` fix is working around also pollutes
naive sequential in-process measurement), then solves for the tightest
``(GEOM_OVERHEAD_BYTES, BYTES_PER_COORD)`` pair that stays above every
sample via linear programming (not a least-squares average — an actual
upper-bound fit), then pads the LP solution by ``PAD_FACTOR`` for safety
against unsampled partitions. Re-run it (``python -m
laurel.utils.dask_sizeof``) whenever targeting a new machine, dataset, or
shapely/geopandas version, pointing it at real sample files for that
context, and update the hard-coded constants below from its output.

Import this module wherever the registrations need to be active. It has an
import-time side effect (the ``@sizeof.register`` decorators).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely
from dask.sizeof import sizeof
from geopandas.array import GeometryArray

# Calibrated against real project data via calibrate_from_real_data() on
# Sherlock (shapely 2.1.1 / geopandas 1.1.1): parking.gpkg (2783 Points) plus
# 30 random real trips_routed partitions, measured in isolated subprocesses.
# LP-fit tight upper bound was GEOM_OVERHEAD_BYTES=7006, BYTES_PER_COORD=100.9;
# values below are that fit padded by PAD_FACTOR=1.3x. Verified: estimate
# exceeded actual RSS delta on 31/31 real samples (ratio 1.3-2.9x). Re-run
# calibrate_from_real_data() and update these if the data or environment
# changes meaningfully (see module docstring).
BYTES_PER_COORD = 131.1
GEOM_OVERHEAD_BYTES = 9108
POINTER_BYTES = 8  # exact: one slot in the GeometryArray's object ndarray


def _coord_upper_bound(shapely_object_array: np.ndarray, n: int) -> int:
    """Estimate an upper-bound byte size for an array of shapely geometries.

    Sums vertex counts across the whole array with a single vectorized
    call to :func:`shapely.get_num_coordinates`, then applies the
    calibrated per-coordinate and per-geometry constants. Works for any
    mix of geometry types (Point, LineString, Polygon, Multi*), since
    vertex count naturally sums across rings/parts.

    Args:
        shapely_object_array: A numpy object array of shapely geometries,
            e.g. ``GeometryArray._data`` or ``GeoSeries.values._data``.
        n: Number of geometries in the array (``len(shapely_object_array)``
            is equivalent, but callers already have this value on hand).

    Returns:
        Estimated size in bytes, biased to overestimate.
    """
    total_coords = int(shapely.get_num_coordinates(shapely_object_array).sum())
    return n * (GEOM_OVERHEAD_BYTES + POINTER_BYTES) + total_coords * BYTES_PER_COORD


@sizeof.register(shapely.Geometry)
def sizeof_shapely_geometry(geom: shapely.Geometry) -> int:
    """Estimate an upper-bound byte size for a single shapely geometry.

    Replaces Dask's default handler, which falls through to
    ``sys.getsizeof`` and returns a constant wrapper size regardless of
    how many coordinates the geometry holds (e.g. the same ~56 bytes for
    a 3-point and a 100,000-point LineString).

    Args:
        geom: Any shapely geometry (covers all geometry types via the
            shared ``shapely.Geometry`` C base class).

    Returns:
        Estimated size in bytes, biased to overestimate.
    """
    return GEOM_OVERHEAD_BYTES + int(shapely.get_num_coordinates(geom)) * BYTES_PER_COORD


@sizeof.register(GeometryArray)
def sizeof_geometry_array(arr: GeometryArray) -> int:
    """Estimate an upper-bound byte size for a GeometryArray.

    Args:
        arr: The extension array backing a GeoSeries column.

    Returns:
        Estimated size in bytes, biased to overestimate.
    """
    return _coord_upper_bound(arr._data, len(arr))


@sizeof.register(gpd.GeoSeries)
def sizeof_geoseries(gs: gpd.GeoSeries) -> int:
    """Estimate an upper-bound byte size for a GeoSeries.

    Replaces Dask's default ``pd.Series`` handler, which GeoSeries would
    otherwise inherit via MRO and which under-counts geometry data (see
    module docstring).

    Args:
        gs: The GeoSeries to size.

    Returns:
        Estimated size in bytes: the index's size plus a coordinate-based
        upper bound for the geometries, biased to overestimate.
    """
    return sizeof(gs.index) + _coord_upper_bound(gs.values._data, len(gs))


@sizeof.register(gpd.GeoDataFrame)
def sizeof_geodataframe(gdf: gpd.GeoDataFrame) -> int:
    """Estimate an upper-bound byte size for a GeoDataFrame.

    Mirrors Dask's default ``pd.DataFrame`` handler column-by-column, but
    routes any geometry-dtype column through the coordinate-based upper
    bound instead of ``memory_usage(deep=False)``. Non-geometry columns
    keep using ``memory_usage(deep=True)`` as before.

    Args:
        gdf: The GeoDataFrame to size.

    Returns:
        Estimated size in bytes, biased to overestimate.
    """
    total = sizeof(gdf.index) + sizeof(gdf.columns)
    for col in gdf._series.values():
        if isinstance(col.dtype, gpd.array.GeometryDtype):
            total += _coord_upper_bound(col.values._data, len(col))
        else:
            total += col.memory_usage(index=False, deep=True)
    return max(1200, total)


def _measure_geometry_file_in_subprocess(path: str, queue) -> None:
    """Load ``path`` and report ``(n_geoms, total_coords, rss_delta)``.

    Runs as the target of a fresh ``multiprocessing`` process so its
    allocator state can't be polluted by (and can't pollute) any other
    measurement -- see :func:`calibrate_from_real_data` for why that
    matters.
    """
    import gc

    import psutil

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    gdf = gpd.read_parquet(path) if path.endswith(".parquet") else gpd.read_file(path)
    geom = gdf.geometry.copy()
    del gdf
    gc.collect()
    rss_after = proc.memory_info().rss
    n = len(geom)
    total_coords = int(shapely.get_num_coordinates(geom.values._data).sum())
    queue.put((n, total_coords, rss_after - rss_before))


def calibrate_from_real_data(
    sample_paths: list[str], pad_factor: float = 1.3
) -> tuple[float, float]:
    """Derive a verified-safe ``(GEOM_OVERHEAD_BYTES, BYTES_PER_COORD)`` pair.

    Measures each path in ``sample_paths`` (GeoParquet or any
    ``geopandas.read_file``-readable format) in its own fresh subprocess,
    then solves for the tightest ``a, b >= 0`` such that
    ``a * n_i + b * coords_i >= actual_bytes_i`` for every sample ``i``
    (a genuine upper-bound fit via linear programming, not a least-squares
    average), then pads that fit by ``pad_factor``.

    Include at least one geometry-simple (point-heavy) and one
    geometry-complex (many-vertex) sample -- a fit derived from only one
    regime silently zeroes out the term that regime doesn't need, which
    then catastrophically underestimates the other regime. See the module
    docstring for a concrete example of exactly this happening.

    Args:
        sample_paths: Real data files to calibrate against. Prefer a random
            sample of actual production partitions plus any small
            reference datasets (e.g. `parking.gpkg`) that appear in the
            same pipeline, so both cost regimes are represented.
        pad_factor: Safety multiplier applied to the LP-fit values before
            returning, to guard against partitions/environments not in the
            sample.

    Returns:
        ``(geom_overhead_bytes, bytes_per_coord)``, already padded -- copy
        these into the ``GEOM_OVERHEAD_BYTES``/``BYTES_PER_COORD`` constants
        above (in that order) if they exceed the current values.
    """
    import multiprocessing as mp

    from scipy.optimize import linprog

    ctx = mp.get_context("spawn")
    rows = []
    for path in sample_paths:
        queue = ctx.Queue()
        proc = ctx.Process(target=_measure_geometry_file_in_subprocess, args=(path, queue))
        proc.start()
        n, total_coords, delta = queue.get()
        proc.join()
        rows.append((n, total_coords, delta))
        print(f"{path}: n={n} coords={total_coords} delta={delta / 1e6:.1f}MB bytes/coord={delta / max(total_coords, 1):.2f}")

    ns = np.array([r[0] for r in rows], dtype=float)
    coords = np.array([r[1] for r in rows], dtype=float)
    actual = np.array([r[2] for r in rows], dtype=float)

    # minimize a + b  s.t.  a*n_i + b*coords_i >= actual_i, a,b >= 0
    result = linprog(
        c=[1, 1],
        A_ub=np.column_stack([-ns, -coords]),
        b_ub=-actual,
        bounds=[(0, None), (0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LP calibration failed: {result.message}")

    geom_overhead, bytes_per_coord = result.x * pad_factor
    estimate = geom_overhead * ns + bytes_per_coord * coords
    n_unsafe = int((estimate < actual).sum())
    ratios = estimate / actual
    print(
        f"\nPadded fit: GEOM_OVERHEAD_BYTES={geom_overhead:.0f} BYTES_PER_COORD={bytes_per_coord:.1f}\n"
        f"Verification: {n_unsafe}/{len(rows)} samples underestimated; "
        f"ratio range [{ratios.min():.2f}, {ratios.max():.2f}]"
    )
    return geom_overhead, bytes_per_coord


if __name__ == "__main__":
    import glob
    import os
    import random

    routes_dir = os.environ["SCRATCH"] + "/laurel/data/02_intermediate/trips_routed"
    parks_path = os.environ["SCRATCH"] + "/laurel/data/02_intermediate/parking.gpkg"
    route_files = sorted(glob.glob(f"{routes_dir}/*.parquet"))
    random.seed(7)
    sample = [parks_path] + random.sample(route_files, min(30, len(route_files)))
    calibrate_from_real_data(sample)
