"""Unit tests for the fused optional-stop-trips nodes in compute_routes.

Covers the per-partition fusion introduced to fix the worker memory
crash-loop, plus the divisions-aware concat that replaced the anti-join
dedup pattern:

- ``_build_partition_trips``: dropna/column-drop, spatial join against
  ``parks``, distance projection, and endpoint-adjacent stop exclusion --
  all fused into one function, one task per partition.
- ``_describe_partition``: the segment distance/time math and final
  renames that remain in ``describe_optional_stop_trips`` after its
  filter+sort responsibilities moved upstream.
- ``select_trips_to_route``: the to-route/not-to-route split.
- ``index_by_vehicle``: indexing by vehicle ID with known divisions.
- ``concat_optional_stops``: the divisions-aware concat + per-partition
  sort that replaced the anti-join + global set_index dedup pattern.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from laurel.pipelines.compute_routes.nodes import (
    _build_partition_trips,
    _describe_partition,
    concat_optional_stops,
    index_by_vehicle,
    select_trips_to_route,
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

    ``split_trip`` is indexed by ``veh_id`` (not a plain column), matching
    what actually arrives at this stage in production: the index set far
    upstream by ``index_by_vehicle`` and carried through unshuffled by every
    node in between (see ``_build_partition_trips``'s no-shuffle invariant).
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
        ).set_index("veh_id")

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
        """veh_id, the leading trip_id_cols entry, stays the index rather
        than becoming a column -- keep_cols_final lists it because it names
        the original trips schema, but only the remaining columns are
        actually selectable here."""
        result = _describe_partition(split_trip, DESCRIBE_PARAMS)
        expected_cols = [c for c in DESCRIBE_PARAMS["keep_cols_final"] if c != "veh_id"]
        assert list(result.columns) == expected_cols

    def test_index_preserved_not_reset(self, split_trip):
        """veh_id remains the index -- no index-dropping side effect."""
        result = _describe_partition(split_trip, DESCRIBE_PARAMS)
        assert result.index.name == "veh_id"
        assert result.index.tolist() == ["V1", "V1"]


SELECT_PARAMS = {
    "drop_cols": ["veh_type", "vin_gvw"],
    "dist_col": "trip_miles",
    "min_dist_miles": 50,
    "debug_subsample": {"active": False, "frac": 0.01},
}


class TestSelectTripsToRoute:
    """Tests for the to-route/not-to-route split."""

    @pytest.fixture
    def dd(self):
        return pytest.importorskip("dask.dataframe")

    def _trips(self, dd):
        pdf = pd.DataFrame(
            {
                "veh_id": ["V1", "V2", "V3"],
                "trip_miles": [10.0, 60.0, 100.0],
                "veh_type": ["a", "a", "a"],
                "vin_gvw": [1, 1, 1],
            }
        )
        return dd.from_pandas(pdf, npartitions=1).set_index("veh_id", sorted=True)

    def test_splits_by_min_dist_miles(self, dd):
        """V1 (10mi) is too short to route; V2 and V3 (60mi, 100mi) are not."""
        to_route, not_to_route = select_trips_to_route(self._trips(dd), SELECT_PARAMS)
        assert sorted(to_route.compute().index.tolist()) == ["V2", "V3"]
        assert sorted(not_to_route.compute().index.tolist()) == ["V1"]

    def test_drop_cols_removed_from_both(self, dd):
        to_route, not_to_route = select_trips_to_route(self._trips(dd), SELECT_PARAMS)
        assert "veh_type" not in to_route.columns
        assert "veh_type" not in not_to_route.columns

    def test_index_preserved_on_both_outputs(self, dd):
        """Both outputs inherit the input's index for free via .loc[]."""
        to_route, not_to_route = select_trips_to_route(self._trips(dd), SELECT_PARAMS)
        assert to_route.index.name == "veh_id"
        assert not_to_route.index.name == "veh_id"

    def test_debug_subsample_applies_only_to_to_route(self, dd):
        """An active debug subsample shrinks the to-route side but leaves
        not_to_route (not part of the expensive routing path) untouched."""
        params = {**SELECT_PARAMS, "debug_subsample": {"active": True, "frac": 0.0}}
        to_route, not_to_route = select_trips_to_route(self._trips(dd), params)
        assert len(to_route.compute()) == 0
        assert len(not_to_route.compute()) == 1


INDEX_PARAMS = {"id_col": "veh_id", "n_partitions": 3}


class TestIndexByVehicle:
    """Tests for indexing a trips DataFrame by vehicle ID with known divisions."""

    @pytest.fixture
    def dd(self):
        return pytest.importorskip("dask.dataframe")

    def _trips_many_vehicles(self, dd, n_vehicles=200, npartitions=4):
        """Enough distinct veh_id values that set_index's quantile-based
        partitioning can actually honor a requested npartitions."""
        pdf = pd.DataFrame(
            {
                "veh_id": [f"V{i:04d}" for i in range(n_vehicles)],
                "some_col": list(range(n_vehicles)),
            }
        )
        return dd.from_pandas(pdf, npartitions=npartitions)

    def test_indexes_by_id_col(self, dd):
        result = index_by_vehicle(self._trips_many_vehicles(dd), INDEX_PARAMS)
        assert result.index.name == "veh_id"
        assert "veh_id" not in result.columns

    def test_known_divisions(self, dd):
        result = index_by_vehicle(self._trips_many_vehicles(dd), INDEX_PARAMS)
        assert result.known_divisions

    def test_output_partition_count_matches_n_partitions(self, dd):
        result = index_by_vehicle(self._trips_many_vehicles(dd), INDEX_PARAMS)
        assert result.npartitions == INDEX_PARAMS["n_partitions"]

    def test_data_preserved(self, dd):
        result = index_by_vehicle(self._trips_many_vehicles(dd), INDEX_PARAMS).compute()
        assert sorted(result["some_col"].tolist()) == list(range(200))


CONCAT_PARAMS = {
    "trip_id_cols": ["veh_id", "end_timestamp_utc"],
}


class TestConcatOptionalStops:
    """Tests for the divisions-aware concat + per-partition sort logic.

    Both inputs are expected to already be disjoint (by construction, via
    ``select_trips_to_route``) and indexed by vehicle ID with known
    divisions (via ``index_by_vehicle``) -- no anti-join/dedup is needed.
    """

    @pytest.fixture
    def dd(self):
        dask_dataframe = pytest.importorskip("dask.dataframe")
        return dask_dataframe

    def _trips_not_to_route(self, dd, npartitions=1):
        pdf = pd.DataFrame(
            {
                "veh_id": ["V1", "V3"],
                "end_timestamp_utc": [T0, T0],
                "some_col": [1, 3],
            }
        )
        return dd.from_pandas(pdf, npartitions=npartitions).set_index("veh_id", sorted=True)

    def _trips_opt(self, dd):
        # V2 is a split trip: two sub-rows with distinct (post-describe)
        # end_timestamp_utc values, deliberately out of chronological order.
        # V4 is a routed-but-unsplit trip, appearing as a single row.
        pdf = pd.DataFrame(
            {
                "veh_id": ["V2", "V2", "V4"],
                "end_timestamp_utc": [T0 + pd.Timedelta(minutes=15), T0, T0],
                "some_col": [21, 20, 40],
            }
        )
        return dd.from_pandas(pdf, npartitions=1).set_index("veh_id", sorted=True)

    def test_split_trip_rows_present(self, dd):
        """Both of V2's split rows (from trips_opt) appear in the output."""
        result = concat_optional_stops(
            self._trips_not_to_route(dd), self._trips_opt(dd), CONCAT_PARAMS
        ).compute()
        assert list(result.index).count("V2") == 2
        assert sorted(result.loc["V2", "some_col"].tolist()) == [20, 21]

    def test_not_to_route_trips_preserved(self, dd):
        """V1 and V3, too short to route, survive unchanged."""
        result = concat_optional_stops(
            self._trips_not_to_route(dd), self._trips_opt(dd), CONCAT_PARAMS
        ).compute()
        assert result.loc["V1", "some_col"] == 1
        assert result.loc["V3", "some_col"] == 3

    def test_unsplit_routed_trip_preserved(self, dd):
        """V4, routed but never split, appears once via trips_opt."""
        result = concat_optional_stops(
            self._trips_not_to_route(dd), self._trips_opt(dd), CONCAT_PARAMS
        ).compute()
        assert result.loc["V4", "some_col"] == 40

    def test_indexed_by_leading_trip_id_col(self, dd):
        """Output is indexed by veh_id rather than carrying it as a column."""
        result = concat_optional_stops(
            self._trips_not_to_route(dd), self._trips_opt(dd), CONCAT_PARAMS
        ).compute()
        assert result.index.name == "veh_id"
        assert "veh_id" not in result.columns

    def test_globally_sorted_by_index(self, dd):
        result = concat_optional_stops(
            self._trips_not_to_route(dd), self._trips_opt(dd), CONCAT_PARAMS
        ).compute()
        assert list(result.index) == sorted(result.index)

    def test_v2_rows_sorted_chronologically_within_vehicle(self, dd):
        """The final per-partition sort orders V2's two rows by
        end_timestamp_utc even though trips_opt handed them in reverse."""
        result = concat_optional_stops(
            self._trips_not_to_route(dd), self._trips_opt(dd), CONCAT_PARAMS
        ).compute()
        assert result.loc["V2", "some_col"].tolist() == [20, 21]

    def test_works_when_not_to_route_spans_multiple_partitions(self, dd):
        result = concat_optional_stops(
            self._trips_not_to_route(dd, npartitions=2), self._trips_opt(dd), CONCAT_PARAMS
        ).compute()
        assert sorted(result.index.tolist()) == ["V1", "V2", "V2", "V3", "V4"]
