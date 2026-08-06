"""Unit tests for the fused optional-stop-trips nodes in compute_routes.

Covers the per-partition fusion introduced to fix the worker memory
crash-loop:

- ``_build_partition_trips``: dropna/column-drop, spatial join against
  ``parks``, distance projection, and endpoint-adjacent stop exclusion --
  all fused into one function, one task per partition.
- ``_describe_partition``: the segment distance/time math and final
  renames that remain in ``describe_optional_stop_trips`` after its
  filter+sort responsibilities moved upstream.
- ``concat_optional_stops``: the anti-join + set_index dedup/sort that
  replaced the old compute+concat+drop_duplicates pattern.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from laurel.pipelines.compute_routes.nodes import (
    _build_partition_trips,
    _describe_partition,
    concat_optional_stops,
)
from laurel.utils.geo import METERS_PER_MILE

CRS = "EPSG:5072"

BUILD_PARAMS = {
    "columns": {
        "route_geom": "route_geom",
        "park_point": "park_point",
        "hex_park": "hex_park",
        "hex_end": "hex_end",
        "dist_along_miles": "dist_along_miles",
        "is_optional": "is_optional",
    },
    "projected_crs": CRS,
    "park_buffer_miles": 1.0,
    "drop_cols_initial": ["orig_point"],
}

TRIP_LEN_MILES = 10.0
TRIP_LEN_M = TRIP_LEN_MILES * METERS_PER_MILE


def _make_parks(offsets_miles: list[float], hex_ids: list[str]) -> gpd.GeoDataFrame:
    """Build a parks GeoDataFrame with an active buffer geometry (for the
    join predicate) and a separate ``park_point`` column (for projection),
    matching what ``get_optional_stop_trips`` hands to the fused function.
    """
    points = [Point(m * METERS_PER_MILE, 0) for m in offsets_miles]
    buffers = [p.buffer(BUILD_PARAMS["park_buffer_miles"] * METERS_PER_MILE) for p in points]
    return gpd.GeoDataFrame(
        {
            "hex_park": hex_ids,
            "park_point": gpd.GeoSeries(points, crs=CRS),
        },
        geometry=gpd.GeoSeries(buffers, crs=CRS),
        crs=CRS,
    )


def _make_part(rows: list[dict]) -> gpd.GeoDataFrame:
    """Build a ``routes``-partition-shaped GeoDataFrame from row dicts.

    Each row dict may set ``route_geom`` (a LineString or None) and
    ``trip_miles_route``; ``orig_point`` (the drop_cols_initial column) is
    filled in automatically.
    """
    df = pd.DataFrame(rows)
    if "orig_point" not in df.columns:
        df["orig_point"] = [Point(0, 0)] * len(df)
    return gpd.GeoDataFrame(df, geometry="route_geom", crs=CRS)


class TestBuildPartitionTrips:
    """Tests for ``_build_partition_trips``."""

    def test_drops_null_geometry_routes(self):
        """A row with no route geometry is dropped before any join happens."""
        part = _make_part(
            [
                {"route_geom": None, "trip_miles_route": 5.0},
                {
                    "route_geom": LineString([(0, 0), (TRIP_LEN_M, 0)]),
                    "trip_miles_route": TRIP_LEN_MILES,
                },
            ]
        )
        parks = _make_parks([], [])
        result = _build_partition_trips(part, parks, BUILD_PARAMS)
        assert len(result) == 1
        assert result["trip_miles_route"].iloc[0] == TRIP_LEN_MILES

    def test_drop_cols_initial_removed(self):
        """Columns named in drop_cols_initial never appear in the output."""
        part = _make_part(
            [{"route_geom": LineString([(0, 0), (TRIP_LEN_M, 0)]), "trip_miles_route": TRIP_LEN_MILES}]
        )
        parks = _make_parks([], [])
        result = _build_partition_trips(part, parks, BUILD_PARAMS)
        assert "orig_point" not in result.columns

    def test_original_row_present_and_unflagged(self):
        """Every input trip contributes exactly one is_optional=False row
        whose dist_along_miles equals the full route length."""
        part = _make_part(
            [{"route_geom": LineString([(0, 0), (TRIP_LEN_M, 0)]), "trip_miles_route": TRIP_LEN_MILES}]
        )
        parks = _make_parks([], [])
        result = _build_partition_trips(part, parks, BUILD_PARAMS)
        orig_rows = result.loc[~result["is_optional"]]
        assert len(orig_rows) == 1
        assert orig_rows["dist_along_miles"].iloc[0] == TRIP_LEN_MILES

    def test_middle_stop_kept_endpoint_stops_excluded(self):
        """Of three joined stops (near start, middle, near end), only the
        middle one survives the endpoint-buffer exclusion filter."""
        part = _make_part(
            [{"route_geom": LineString([(0, 0), (TRIP_LEN_M, 0)]), "trip_miles_route": TRIP_LEN_MILES}]
        )
        parks = _make_parks([0.31, 5.0, 9.32], ["H_START", "H_MID", "H_END"])
        result = _build_partition_trips(part, parks, BUILD_PARAMS)
        opt_rows = result.loc[result["is_optional"]]
        assert list(opt_rows["hex_end"]) == ["H_MID"]
        assert opt_rows["dist_along_miles"].iloc[0] == pytest.approx(5.0, abs=1e-6)

    def test_no_matching_stops_yields_only_original_row(self):
        """A route that intersects no park buffer produces no optional rows."""
        part = _make_part(
            [{"route_geom": LineString([(0, 0), (TRIP_LEN_M, 0)]), "trip_miles_route": TRIP_LEN_MILES}]
        )
        parks = _make_parks([100.0], ["H_FAR"])  # 100 miles away, never intersects
        result = _build_partition_trips(part, parks, BUILD_PARAMS)
        assert len(result) == 1
        assert not result["is_optional"].any()

    def test_route_geom_and_park_point_dropped_from_output(self):
        """Geometry columns are dropped before returning (memory-saving intent)."""
        part = _make_part(
            [{"route_geom": LineString([(0, 0), (TRIP_LEN_M, 0)]), "trip_miles_route": TRIP_LEN_MILES}]
        )
        parks = _make_parks([5.0], ["H_MID"])
        result = _build_partition_trips(part, parks, BUILD_PARAMS)
        assert "route_geom" not in result.columns
        assert "park_point" not in result.columns

    def test_input_part_not_mutated(self):
        """The raw input partition retains its original columns/rows after the call."""
        part = _make_part(
            [{"route_geom": LineString([(0, 0), (TRIP_LEN_M, 0)]), "trip_miles_route": TRIP_LEN_MILES}]
        )
        parks = _make_parks([5.0], ["H_MID"])
        n_cols_before = len(part.columns)
        _build_partition_trips(part, parks, BUILD_PARAMS)
        assert len(part.columns) == n_cols_before
        assert "route_geom" in part.columns


DESCRIBE_PARAMS = {
    "columns": {
        "dist_along_miles": "dist_along_miles",
        "speed_route": "speed_route",
        "hours_orig": "hours_orig",
        "hours_route": "hours_route",
        "start_time": "start_time",
    },
    "trip_id_cols": ["veh_id", "end_timestamp_utc"],
    "rename_cols_final": {
        "start_time": "new_start",
        "end_time": "new_end",
        "trip_miles": "trip_miles_route_seg",
        "trip_hrs": "trip_hrs_route_seg",
    },
    "keep_cols_final": [
        "veh_id",
        "end_timestamp_utc",
        "start_time",
        "end_time",
        "trip_miles",
        "trip_hrs",
    ],
}

T0 = pd.Timestamp("2024-01-01 00:00:00")


class TestDescribePartition:
    """Tests for ``_describe_partition``, using one trip split into two
    sub-rows (an is_optional=True stop at mile 5, and the original
    is_optional=False row at mile 10) as already produced and filtered by
    ``_build_partition_trips``.
    """

    @pytest.fixture
    def split_trip(self) -> pd.DataFrame:
        # Route computed 10 miles at 50 mph = 0.2h; the observed trip took
        # 0.25h (hours_orig), so the proportional time_scaler is 1.25.
        return pd.DataFrame(
            {
                "veh_id": ["V1", "V1"],
                "end_timestamp_utc": [T0, T0],
                "dist_along_miles": [10.0, 5.0],  # deliberately unsorted
                "speed_route": [50.0, 50.0],
                "hours_orig": [0.25, 0.25],
                "hours_route": [0.2, 0.2],
                "start_time": [T0, T0],
            }
        )

    def test_sorts_by_trip_and_distance(self, split_trip):
        """The mile-5 segment (lower dist_along_miles) sorts before mile-10."""
        result = _describe_partition(split_trip, DESCRIBE_PARAMS)
        expected_first_end = T0 + pd.Timedelta(minutes=7.5)
        assert result["start_time"].tolist() == [T0, expected_first_end]

    def test_segment_times_sum_to_observed_hours(self, split_trip):
        """Proportional time allocation preserves the total observed trip time."""
        result = _describe_partition(split_trip, DESCRIBE_PARAMS)
        last_new_end = (T0 + pd.to_timedelta(0.25, unit="h")).round("s")
        assert result["end_time"].iloc[-1] == last_new_end

    def test_new_start_chains_from_previous_new_end(self, split_trip):
        """The second sub-trip's start equals the first sub-trip's end."""
        result = _describe_partition(split_trip, DESCRIBE_PARAMS)
        assert result["end_time"].iloc[0] == result["start_time"].iloc[1]

    def test_keep_cols_final_applied(self, split_trip):
        result = _describe_partition(split_trip, DESCRIBE_PARAMS)
        assert list(result.columns) == DESCRIBE_PARAMS["keep_cols_final"]


CONCAT_PARAMS = {
    "drop_cols": ["veh_type", "vin_gvw"],
    "trip_id_cols": ["veh_id", "end_timestamp_utc"],
    "n_partitions": 2,
}


class TestConcatOptionalStops:
    """Tests for the anti-join + set_index dedup/sort logic."""

    @pytest.fixture
    def dd(self):
        dask_dataframe = pytest.importorskip("dask.dataframe")
        return dask_dataframe

    def _trips_orig(self, dd, npartitions=1):
        pdf = pd.DataFrame(
            {
                "veh_id": ["V1", "V2", "V3"],
                "end_timestamp_utc": [T0, T0, T0],
                "veh_type": ["a", "a", "a"],
                "vin_gvw": [1, 1, 1],
                "some_col": [1, 2, 3],
            }
        )
        return dd.from_pandas(pdf, npartitions=npartitions)

    def _trips_opt(self, dd):
        pdf = pd.DataFrame(
            {
                "veh_id": ["V2", "V2"],
                "end_timestamp_utc": [T0, T0],
                "some_col": [20, 21],
            }
        )
        return dd.from_pandas(pdf, npartitions=1)

    def test_split_trip_original_row_dropped(self, dd):
        """V2's original row is replaced entirely by its two split rows."""
        result = concat_optional_stops(self._trips_orig(dd), self._trips_opt(dd), CONCAT_PARAMS).compute()
        assert list(result.index).count("V2") == 2
        assert sorted(result.loc["V2", "some_col"].tolist()) == [20, 21]

    def test_unsplit_trips_preserved(self, dd):
        """V1 and V3, which never appear in trips_opt, survive unchanged."""
        result = concat_optional_stops(self._trips_orig(dd), self._trips_opt(dd), CONCAT_PARAMS).compute()
        assert result.loc["V1", "some_col"] == 1
        assert result.loc["V3", "some_col"] == 3

    def test_drop_cols_removed(self, dd):
        result = concat_optional_stops(self._trips_orig(dd), self._trips_opt(dd), CONCAT_PARAMS).compute()
        assert "veh_type" not in result.columns
        assert "vin_gvw" not in result.columns

    def test_indexed_by_leading_trip_id_col(self, dd):
        """Output is indexed by veh_id rather than carrying it as a column."""
        result = concat_optional_stops(self._trips_orig(dd), self._trips_opt(dd), CONCAT_PARAMS).compute()
        assert result.index.name == "veh_id"
        assert "veh_id" not in result.columns

    def test_globally_sorted_by_index(self, dd):
        result = concat_optional_stops(self._trips_orig(dd), self._trips_opt(dd), CONCAT_PARAMS).compute()
        assert list(result.index) == sorted(result.index)

    def test_anti_join_works_across_multiple_partitions(self, dd):
        """The dedup is correct even when trips_orig spans multiple partitions."""
        result = concat_optional_stops(
            self._trips_orig(dd, npartitions=3), self._trips_opt(dd), CONCAT_PARAMS
        ).compute()
        assert sorted(result.index.tolist()) == ["V1", "V2", "V2", "V3"]

    def _trips_orig_many_vehicles(self, dd, n_vehicles=200, npartitions=4):
        """A larger fixture with enough distinct veh_id values that
        set_index's quantile-based partitioning can actually honor a
        requested npartitions (a handful of distinct index values, as in
        the other tests' 3-row fixture, isn't enough for that)."""
        pdf = pd.DataFrame(
            {
                "veh_id": [f"V{i:04d}" for i in range(n_vehicles)],
                "end_timestamp_utc": [T0] * n_vehicles,
                "veh_type": ["a"] * n_vehicles,
                "vin_gvw": [1] * n_vehicles,
                "some_col": list(range(n_vehicles)),
            }
        )
        return dd.from_pandas(pdf, npartitions=npartitions)

    def test_output_partition_count_matches_n_partitions(self, dd):
        """set_index's npartitions kwarg controls the output partition count directly."""
        params = {**CONCAT_PARAMS, "n_partitions": 5}
        result = concat_optional_stops(
            self._trips_orig_many_vehicles(dd), self._trips_opt(dd), params
        )
        assert result.npartitions == 5
