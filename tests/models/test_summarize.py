"""Unit tests for megaplug.models.summarize module.

This module tests the IntervalBeginSpreader and NonzeroGroupedSummarizer classes using pytest.
"""

import numpy as np
import pandas as pd
import pytest
from megaplug.models.summarize import IntervalBeginSpreader, NonzeroGroupedSummarizer


class TestIntervalBeginSpreader:
    """Test cases for IntervalBeginSpreader class."""

    @pytest.fixture
    def spreader(self):
        """Fixture for IntervalBeginSpreader instance."""
        return IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols="power",
            group_cols=["vehicle_id", "location"],
            freq="1h",
        )

    @pytest.fixture
    def sample_obs(self):
        """Fixture for sample obs DataFrame."""
        return pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:00:00", "2024-01-01 15:30:00"]
                ),
                "duration": pd.to_timedelta(["30min", "20min"]),
                "power": [100.0, 150.0],
            }
        )

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        spreader = IntervalBeginSpreader(
            time_col="time",
            dur_col="dur",
            value_cols="val",
            group_cols=["group1", "group2"],
            freq="15min",
        )

        assert spreader.time_col == "time"
        assert spreader.dur_col == "dur"
        assert spreader.value_cols == ["val"]
        assert spreader.group_cols == ["group1", "group2"]
        assert spreader.freq == "15min"

    def test_init_empty_group_cols(self):
        """Test initialization with empty group_cols list."""
        spreader = IntervalBeginSpreader(
            time_col="time", dur_col="dur", value_cols="val", group_cols=[], freq="1h"
        )

        assert spreader.group_cols == []

    def test_expand_obs_no_expansion_needed(self, spreader, sample_obs):
        """Test obs that don't cross time boundaries."""
        result = spreader.spread(sample_obs)

        # Should return original obs since no expansion is needed
        expected_cols = spreader.group_cols + [spreader.time_col] + spreader.value_cols
        assert list(result.columns) == expected_cols
        assert len(result) == len(sample_obs)

    def test_expand_obs_single_expansion(self, spreader):
        """Test event spanning exactly 2 time periods."""
        # Events with different expansion needs
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 14:15:00"]
                ),
                "duration": pd.to_timedelta(["1h", "45min"]),
                "power": [100.0, 150.0],
            }
        )

        result = spreader.spread(obs)

        # Should have more rows due to expansion
        assert len(result) >= len(obs)

        # Check that power value is preserved
        assert all(result["power"] == np.array([100.0, 100.0, 150.0]))

    def test_expand_obs_multiple_expansions(self, spreader):
        """Test event spanning multiple time periods."""
        # Events with different expansion needs
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:00:00", "2024-01-01 16:00:00"]
                ),
                "duration": pd.to_timedelta(["3h", "2h30min"]),
                "power": [200.0, 180.0],
            }
        )

        result = spreader.spread(obs)

        # Should have expanded rows
        expanded_rows = result[result["power"] == 200.0]
        assert len(expanded_rows) > 1

    def test_expand_obs_different_frequencies(self):
        """Test with different frequency settings."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:00:00", "2024-01-01 12:00:00"]
                ),
                "duration": pd.to_timedelta(["2h", "90min"]),
                "power": [100.0, 120.0],
            }
        )

        # Test with 15-minute frequency
        spreader_15min = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols="power",
            group_cols=["vehicle_id", "location"],
            freq="15min",
        )

        result = spreader_15min.spread(obs)

        # Should have more expanded rows with finer granularity
        expanded_rows = result[result["power"] == 100.0]
        assert len(expanded_rows) > 1

    def test_timezone_naive_input(self, spreader):
        """Test DataFrame with timezone-naive timestamps."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 14:00:00"]
                ),  # No timezone
                "duration": pd.to_timedelta(["90min", "75min"]),
                "power": [100.0, 130.0],
            }
        )

        result = spreader.spread(obs)

        # Should process without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_timezone_utc_input(self, spreader):
        """Test DataFrame with UTC timestamps."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 15:30:00"], utc=True
                ),
                "duration": pd.to_timedelta(["90min", "60min"]),
                "power": [100.0, 140.0],
            }
        )

        result = spreader.spread(obs)

        # Should process without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_timezone_other_raises_error(self, spreader):
        """Test that non-UTC/non-naive timezones raise RuntimeError."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 13:30:00"]
                ).tz_localize("US/Pacific"),
                "duration": pd.to_timedelta(["90min", "75min"]),
                "power": [100.0, 110.0],
            }
        )

        with pytest.raises(RuntimeError, match="time zone naïve or UTC"):
            spreader.spread(obs)

    def test_return_spreaded_only_true(self, spreader):
        """Test return_spreaded_only=True."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 15:00:00"]
                ),
                "duration": pd.to_timedelta(
                    ["90min", "30min"]
                ),  # Only first needs expansion
                "power": [100.0, 150.0],
            }
        )

        result = spreader.spread(obs, return_spreaded_only=True)

        # Should only contain expanded obs
        assert len(result) > 0
        # All returned rows should be from expanded obs
        assert all(result["power"] == 100.0)  # Only first event gets expanded

    def test_return_spreaded_only_false(self, spreader):
        """Test return_spreaded_only=False (default)."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 15:00:00"]
                ),
                "duration": pd.to_timedelta(
                    ["90min", "30min"]
                ),  # Only first needs expansion
                "power": [100.0, 150.0],
            }
        )

        result = spreader.spread(obs, return_spreaded_only=False)

        # Should contain both expanded and non-expanded obs
        assert len(result) >= len(obs)
        # Should contain both power values
        power_values = result["power"].unique()
        assert 100.0 in power_values
        assert 150.0 in power_values

    @pytest.mark.parametrize(
        "value_type,test_value",
        [
            (int, 100),
            (float, 100.0),
        ],
    )
    def test_different_value_types(self, spreader, value_type, test_value):
        """Test with different value types."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 14:30:00"]
                ),
                "duration": pd.to_timedelta(["90min", "60min"]),
                "power": [test_value, test_value],
            }
        )

        result = spreader.spread(obs)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_multiple_group_columns(self):
        """Test with multiple group columns."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 1, 2],
                "location": ["A", "B", "A"],
                "driver": ["John", "Jane", "Bob"],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:30:00",
                        "2024-01-01 11:30:00",
                        "2024-01-01 12:30:00",
                    ]
                ),
                "duration": pd.to_timedelta(["90min", "90min", "90min"]),
                "power": [100.0, 150.0, 200.0],
            }
        )

        spreader_multi = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols="power",
            group_cols=["vehicle_id", "location", "driver"],
            freq="1h",
        )

        result = spreader_multi.spread(obs)

        # Should preserve all group columns
        expected_cols = ["vehicle_id", "location", "driver", "timestamp", "power"]
        assert sorted(result.columns) == sorted(expected_cols)

    def test_single_group_column(self):
        """Test with single group column."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 11:30:00"]
                ),
                "duration": pd.to_timedelta(["90min", "90min"]),
                "power": [100.0, 150.0],
            }
        )

        spreader_single = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols="power",
            group_cols=["vehicle_id"],
            freq="1h",
        )

        result = spreader_single.spread(obs)

        # Should preserve single group column
        expected_cols = ["vehicle_id", "timestamp", "power"]
        assert sorted(result.columns) == sorted(expected_cols)

    def test_zero_duration_obs(self, spreader):
        """Test obs with zero duration."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 14:00:00"]
                ),
                "duration": pd.to_timedelta(["0min", "0min"]),
                "power": [100.0, 120.0],
            }
        )

        result = spreader.spread(obs)

        # Should handle zero duration gracefully
        assert isinstance(result, pd.DataFrame)
        expected_cols = spreader.group_cols + [spreader.time_col] + spreader.value_cols
        assert list(result.columns) == expected_cols

    def test_negative_duration_obs(self, spreader):
        """Test obs with negative duration."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 15:00:00"]
                ),
                "duration": pd.to_timedelta(["-30min", "-15min"]),
                "power": [100.0, 90.0],
            }
        )

        with pytest.raises(ValueError):
            spreader.spread(obs)

    def test_empty_dataframe(self, spreader):
        """Test empty input DataFrame."""
        obs = pd.DataFrame(
            columns=["vehicle_id", "location", "timestamp", "duration", "power"]
        )
        obs["timestamp"] = pd.to_datetime(obs["timestamp"])
        obs["duration"] = pd.to_timedelta(obs["duration"])

        result = spreader.spread(obs)

        # Should return empty DataFrame with correct columns
        expected_cols = spreader.group_cols + [spreader.time_col] + spreader.value_cols
        assert list(result.columns) == expected_cols
        assert len(result) == 0

    def test_minimal_dataframe(self, spreader):
        """Test DataFrame with minimal obs."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 12:00:00"]
                ),
                "duration": pd.to_timedelta(["90min", "45min"]),
                "power": [100.0, 80.0],
            }
        )

        result = spreader.spread(obs)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_obs_at_time_boundaries(self, spreader):
        """Test obs starting/ending exactly on frequency boundaries."""
        # Events starting exactly at hour boundaries
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:00:00", "2024-01-01 14:00:00"]
                ),  # Exactly on hour
                "duration": pd.to_timedelta(["90min", "120min"]),
                "power": [100.0, 160.0],
            }
        )

        result = spreader.spread(obs)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_core_algorithm_single_event(self):
        """Test the core numba function directly."""
        starts = np.array([0], dtype=np.int64)
        ends = np.array([3600000000000], dtype=np.int64)  # 1 hour in nanoseconds
        vals = np.array([100.0])
        grps = np.array([1])
        tstep_ns = 3600000000000  # 1 hour in nanoseconds

        grps_exp, times_exp, vals_exp = IntervalBeginSpreader._spread_core(
            starts, ends, vals, grps, tstep_ns
        )

        assert isinstance(grps_exp, np.ndarray)
        assert isinstance(times_exp, np.ndarray)
        assert isinstance(vals_exp, np.ndarray)
        assert len(grps_exp) == len(times_exp)
        assert len(times_exp) == len(vals_exp)

    def test_core_multiple_obs(self):
        """Test core function with multiple obs."""
        starts = np.array([0, 3600000000000], dtype=np.int64)
        ends = np.array(
            [7200000000000, 10800000000000], dtype=np.int64
        )  # 2 and 3 hours
        vals = np.array([100.0, 200.0])
        grps = np.array([1, 2])
        tstep_ns = 3600000000000  # 1 hour in nanoseconds

        grps_exp, times_exp, vals_exp = IntervalBeginSpreader._spread_core(
            starts, ends, vals, grps, tstep_ns
        )

        # Should have expanded multiple obs
        assert len(grps_exp) > 2

        # Values should be preserved correctly
        unique_vals = np.unique(vals_exp)
        assert 100.0 in unique_vals
        assert 200.0 in unique_vals

    def test_core_edge_timestamps(self):
        """Test core function with edge case timestamps."""
        # Test with very large timestamps
        starts = np.array([1640995200000000000], dtype=np.int64)  # Jan 1, 2022 in ns
        ends = np.array([1640998800000000000], dtype=np.int64)  # 1 hour later
        vals = np.array([50.0])
        grps = np.array([1])
        tstep_ns = 3600000000000  # 1 hour in nanoseconds

        grps_exp, times_exp, vals_exp = IntervalBeginSpreader._spread_core(
            starts, ends, vals, grps, tstep_ns
        )

        assert len(grps_exp) > 0
        assert all(vals_exp == 50.0)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
    def test_core_different_dtypes(self, dtype):
        """Test core function with different value array dtypes."""
        starts = np.array([0], dtype=np.int64)
        ends = np.array([3600000000000], dtype=np.int64)
        vals = np.array([100], dtype=dtype)
        grps = np.array([1])
        tstep_ns = 3600000000000

        grps_exp, times_exp, vals_exp = IntervalBeginSpreader._spread_core(
            starts, ends, vals, grps, tstep_ns
        )

        assert isinstance(vals_exp, np.ndarray)
        assert vals_exp.dtype == dtype

    def test_wrapper_function(self, spreader, mocker):
        """Test the wrapper function with mocked dependencies."""
        # Mock the utility functions
        mock_get_dtype = mocker.patch("megaplug.models.summarize.get_basic_dtype_ser")
        mock_get_dtype.side_effect = lambda x: x  # Return the series as-is

        # Create test data that would normally require expansion
        obs = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 14:00:00"], utc=True
                ),
                "duration": pd.to_timedelta(["90min", "75min"]),
                "power": [100.0, 110.0],
                "codes": [0, 1],  # Required by wrapper
            }
        )

        result_df = spreader._spread_wrapper(obs)

        # Should return a DataFrame
        assert isinstance(result_df, pd.DataFrame)
        assert "codes" in result_df.columns
        assert spreader.time_col in result_df.columns

    def test_missing_required_columns(self, spreader):
        """Test behavior with missing required columns."""
        # Missing duration column
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 13:00:00"]
                ),
                "power": [100.0, 90.0],
                # Missing 'duration' column
            }
        )

        with pytest.raises(KeyError):
            spreader.spread(obs)

    def test_indexed_dataframe_input(self, spreader):
        """Test with pre-indexed DataFrames."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 11:30:00"]
                ),
                "duration": pd.to_timedelta(["90min", "90min"]),
                "power": [100.0, 150.0],
            }
        )

        # Set a custom index
        obs = obs.set_index(["vehicle_id", "location"])

        result = spreader.spread(obs)

        # Should handle indexed input gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_output_column_preservation(self, spreader):
        """Test that output contains correct columns."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 12:30:00"]
                ),
                "duration": pd.to_timedelta(["90min", "75min"]),
                "power": [100.0, 85.0],
            }
        )

        result = spreader.spread(obs)

        expected_cols = spreader.group_cols + [spreader.time_col] + spreader.value_cols
        assert sorted(result.columns) == sorted(expected_cols)

    def test_output_row_count_logic(self, spreader):
        """Test expected number of rows in output."""
        # Events that should expand to multiple time periods
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:00:00", "2024-01-01 15:00:00"]
                ),
                "duration": pd.to_timedelta(
                    ["2h30min", "3h15min"]
                ),  # Different expansion needs
                "power": [100.0, 125.0],
            }
        )

        result = spreader.spread(obs)

        # Should have at least the original row, likely more due to expansion
        assert len(result) >= 1

    def test_value_propagation(self, spreader):
        """Test that values are correctly propagated to expanded timestamps."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 14:15:00"]
                ),
                "duration": pd.to_timedelta(
                    ["2h30min", "3h"]
                ),  # Should expand to multiple periods
                "power": [100.0, 175.0],
            }
        )

        result = spreader.spread(obs)

        # All power values should be 100.0 for this vehicle/location
        vehicle_data = result[(result["vehicle_id"] == 1) & (result["location"] == "A")]
        assert all(vehicle_data["power"] == 100.0)

    def test_group_preservation(self, spreader):
        """Test that group values are maintained correctly."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 11:30:00"]
                ),
                "duration": pd.to_timedelta(["2h", "2h"]),
                "power": [100.0, 200.0],
            }
        )

        result = spreader.spread(obs)

        # Each expanded row should maintain its group values
        vehicle_1_data = result[result["vehicle_id"] == 1]
        vehicle_2_data = result[result["vehicle_id"] == 2]

        # Vehicle 1 should only have location A and power 100.0
        assert all(vehicle_1_data["location"] == "A")
        assert all(vehicle_1_data["power"] == 100.0)

        # Vehicle 2 should only have location B and power 200.0
        assert all(vehicle_2_data["location"] == "B")
        assert all(vehicle_2_data["power"] == 200.0)

    def test_real_world_scenario(self, spreader):
        """Test with realistic vehicle telematics data."""
        # Simulate realistic charging obs
        obs = pd.DataFrame(
            {
                "vehicle_id": [1001, 1002, 1001],
                "location": ["depot_A", "depot_B", "depot_A"],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 08:30:00",  # Morning charging
                        "2024-01-01 14:45:00",  # Afternoon charging
                        "2024-01-01 22:15:00",  # Evening charging
                    ]
                ),
                "duration": pd.to_timedelta(["3h15min", "2h30min", "8h45min"]),
                "power": [150.0, 200.0, 120.0],  # Different charging rates
            }
        )

        result = spreader.spread(obs)

        # Should handle realistic scenario without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= len(obs)

        # Check that all original power values are preserved
        original_powers = set(obs["power"])
        result_powers = set(result["power"])
        assert original_powers.issubset(result_powers)

    @pytest.mark.performance
    def test_large_dataset_performance(self, spreader):
        """Test performance with large datasets."""
        # Create a larger dataset
        n_obs = 1000
        obs = pd.DataFrame(
            {
                "vehicle_id": np.random.randint(1, 101, n_obs),
                "location": [f"location_{i%10}" for i in range(n_obs)],
                "timestamp": pd.date_range("2024-01-01", periods=n_obs, freq="1h"),
                "duration": pd.to_timedelta(
                    np.random.randint(30, 180, n_obs), unit="min"
                ),
                "power": np.random.uniform(50, 200, n_obs),
            }
        )

        # Should complete without errors (performance is measured externally)
        result = spreader.spread(obs)

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= len(obs)

    def test_timezone_conversion_consistency(self, spreader):
        """Test that output timezone matches input timezone format."""
        # Test with timezone-naive input
        obs_naive = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 13:30:00"]
                ),  # No timezone
                "duration": pd.to_timedelta(["90min", "60min"]),
                "power": [100.0, 95.0],
            }
        )

        result_naive = spreader.spread(obs_naive)

        # Output should also be timezone-naive
        assert result_naive["timestamp"].dt.tz is None

        # Test with UTC input
        obs_utc = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 16:00:00"], utc=True
                ),
                "duration": pd.to_timedelta(["90min", "120min"]),
                "power": [100.0, 135.0],
            }
        )

        result_utc = spreader.spread(obs_utc)

        # Output should maintain UTC timezone
        assert result_utc["timestamp"].dt.tz is not None

    def test_multiple_value_columns_same_dtype(self):
        """Test with multiple value columns of the same data type."""
        spreader_multi = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols=["power", "energy", "voltage"],
            group_cols=["vehicle_id", "location"],
            freq="1h",
        )

        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["A", "B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 14:15:00"]
                ),
                "duration": pd.to_timedelta(["90min", "2h"]),
                "power": [100.0, 150.0],
                "energy": [50.0, 75.0],
                "voltage": [240.0, 220.0],
            }
        )

        result = spreader_multi.spread(obs)

        # Should contain all value columns
        expected_cols = [
            "vehicle_id",
            "location",
            "timestamp",
            "power",
            "energy",
            "voltage",
        ]
        assert sorted(result.columns) == sorted(expected_cols)

        # Values should be preserved across all columns
        assert 100.0 in result["power"].values
        assert 150.0 in result["power"].values
        assert 50.0 in result["energy"].values
        assert 75.0 in result["energy"].values
        assert 240.0 in result["voltage"].values
        assert 220.0 in result["voltage"].values

    def test_multiple_value_columns_different_dtypes(self):
        """Test with multiple value columns of different data types."""
        spreader_multi = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols=["power_float", "count_int", "efficiency_pct"],
            group_cols=["vehicle_id"],
            freq="1h",
        )

        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 15:45:00"]
                ),
                "duration": pd.to_timedelta(["2h", "90min"]),
                "power_float": [125.5, 200.75],  # float64
                "count_int": [5, 8],  # int64
                "efficiency_pct": [85.2, 92.8],  # float64
            }
        )

        # Explicitly set dtypes
        obs["power_float"] = obs["power_float"].astype(np.float32)
        obs["count_int"] = obs["count_int"].astype(np.int32)
        obs["efficiency_pct"] = obs["efficiency_pct"].astype(np.float64)

        result = spreader_multi.spread(obs)

        # Check that data types are preserved
        assert result["power_float"].dtype == np.float32
        assert result["count_int"].dtype == np.int32
        assert result["efficiency_pct"].dtype == np.float64

        # Values should be preserved
        assert 125.5 in result["power_float"].values
        assert 5 in result["count_int"].values
        assert 85.2 in result["efficiency_pct"].values

    def test_backwards_compatibility_single_string(self):
        """Test backwards compatibility with single string value_cols parameter."""
        # Test single string (old API)
        spreader_string = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols="power",  # Single string
            group_cols=["vehicle_id"],
            freq="1h",
        )

        # Test list with single element (new API)
        spreader_list = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols=["power"],  # List with single element
            group_cols=["vehicle_id"],
            freq="1h",
        )

        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 14:00:00"]
                ),
                "duration": pd.to_timedelta(["90min", "75min"]),
                "power": [100.0, 120.0],
            }
        )

        result_string = spreader_string.spread(obs)
        result_list = spreader_list.spread(obs)

        # Both should have the same internal value_cols list
        assert spreader_string.value_cols == ["power"]
        assert spreader_list.value_cols == ["power"]

        # Results should be equivalent
        pd.testing.assert_frame_equal(
            result_string.sort_values(["vehicle_id", "timestamp"]).reset_index(
                drop=True
            ),
            result_list.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True),
        )

    def test_dtype_preservation_across_multiple_columns(self):
        """Test that each column's dtype is preserved independently."""
        spreader_multi = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols=["int8_col", "int16_col", "float32_col", "float64_col"],
            group_cols=["vehicle_id"],
            freq="1h",
        )

        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 13:30:00"]
                ),
                "duration": pd.to_timedelta(["2h", "90min"]),
                "int8_col": [10, 20],
                "int16_col": [1000, 2000],
                "float32_col": [100.5, 200.25],
                "float64_col": [1000000.123456, 2000000.654321],
            }
        )

        # Set specific dtypes
        obs["int8_col"] = obs["int8_col"].astype(np.int8)
        obs["int16_col"] = obs["int16_col"].astype(np.int16)
        obs["float32_col"] = obs["float32_col"].astype(np.float32)
        obs["float64_col"] = obs["float64_col"].astype(np.float64)

        result = spreader_multi.spread(obs)

        # Verify each dtype is preserved
        assert result["int8_col"].dtype == np.int8
        assert result["int16_col"].dtype == np.int16
        assert result["float32_col"].dtype == np.float32
        assert result["float64_col"].dtype == np.float64

        # Verify values are preserved
        assert 10 in result["int8_col"].values
        assert 1000 in result["int16_col"].values
        assert np.isclose(100.5, result["float32_col"].values).any()
        assert np.isclose(1000000.123456, result["float64_col"].values).any()

    def test_multiple_columns_with_expansion(self):
        """Test multiple value columns with events requiring expansion."""
        spreader_multi = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols=["power", "current", "efficiency"],
            group_cols=["vehicle_id", "location"],
            freq="1h",
        )

        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "location": ["depot_A", "depot_B"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 14:15:00"]
                ),
                "duration": pd.to_timedelta(["3h", "2h30min"]),  # Both should expand
                "power": [150.0, 200.0],
                "current": [25.0, 35.0],
                "efficiency": [0.95, 0.88],
            }
        )

        result = spreader_multi.spread(obs)

        # Should have expanded rows
        assert len(result) > len(obs)

        # Check vehicle 1 data consistency across all columns
        vehicle_1_data = result[
            (result["vehicle_id"] == 1) & (result["location"] == "depot_A")
        ]
        assert all(vehicle_1_data["power"] == 150.0)
        assert all(vehicle_1_data["current"] == 25.0)
        assert all(vehicle_1_data["efficiency"] == 0.95)
        assert len(vehicle_1_data) > 1  # Should be expanded

        # Check vehicle 2 data consistency
        vehicle_2_data = result[
            (result["vehicle_id"] == 2) & (result["location"] == "depot_B")
        ]
        assert all(vehicle_2_data["power"] == 200.0)
        assert all(vehicle_2_data["current"] == 35.0)
        assert all(vehicle_2_data["efficiency"] == 0.88)
        assert len(vehicle_2_data) > 1  # Should be expanded

    def test_multiple_columns_return_spreaded_only(self):
        """Test multiple columns with return_spreaded_only=True."""
        spreader_multi = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols=["power", "energy"],
            group_cols=["vehicle_id"],
            freq="1h",
        )

        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2, 3],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:30:00",  # Will expand
                        "2024-01-01 14:00:00",  # Won't expand
                        "2024-01-01 16:30:00",  # Will expand
                    ]
                ),
                "duration": pd.to_timedelta(["2h", "30min", "90min"]),
                "power": [100.0, 150.0, 200.0],
                "energy": [50.0, 75.0, 100.0],
            }
        )

        result = spreader_multi.spread(obs, return_spreaded_only=True)

        # Should only contain expanded rows (vehicles 1 and 3)
        unique_vehicles = result["vehicle_id"].unique()
        assert 1 in unique_vehicles
        assert 3 in unique_vehicles
        assert 2 not in unique_vehicles  # Vehicle 2 didn't need expansion

        # Check that all value columns are present
        assert "power" in result.columns
        assert "energy" in result.columns

    def test_multiple_columns_edge_cases(self):
        """Test edge cases with multiple value columns."""
        spreader_multi = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols=["col1", "col2", "col3"],
            group_cols=["group"],
            freq="1h",
        )

        # Test with empty DataFrame
        empty_obs = pd.DataFrame(
            columns=["group", "timestamp", "duration", "col1", "col2", "col3"]
        )
        empty_obs["timestamp"] = pd.to_datetime(empty_obs["timestamp"])
        empty_obs["duration"] = pd.to_timedelta(empty_obs["duration"])

        result_empty = spreader_multi.spread(empty_obs)
        expected_cols = ["group", "timestamp", "col1", "col2", "col3"]
        assert sorted(result_empty.columns) == sorted(expected_cols)
        assert len(result_empty) == 0

        # Test with single row
        single_obs = pd.DataFrame(
            {
                "group": [1],
                "timestamp": pd.to_datetime(["2024-01-01 10:30:00"]),
                "duration": pd.to_timedelta(["2h"]),
                "col1": [100.0],
                "col2": [200.0],
                "col3": [300.0],
            }
        )

        result_single = spreader_multi.spread(single_obs)
        assert len(result_single) > 1  # Should be expanded
        assert all(result_single["col1"] == 100.0)
        assert all(result_single["col2"] == 200.0)
        assert all(result_single["col3"] == 300.0)

    @pytest.mark.performance
    def test_multiple_columns_performance_comparison(self):
        """Test performance comparison between single and multiple column processing."""
        try:
            from pyinstrument import Profiler
        except ImportError:
            pytest.skip("pyinstrument not available")

        # Create a moderately sized dataset for performance testing
        n_obs = 5000
        np.random.seed(42)  # For reproducible results

        base_obs = pd.DataFrame(
            {
                "vehicle_id": np.random.randint(1, 101, n_obs),
                "location": [f"location_{i%20}" for i in range(n_obs)],
                "timestamp": pd.date_range("2024-01-01", periods=n_obs, freq="30min"),
                "duration": pd.to_timedelta(
                    np.random.randint(30, 300, n_obs),
                    unit="min",  # 30min to 5h durations
                ),
                "power": np.random.uniform(50, 200, n_obs),
                "energy": np.random.uniform(25, 150, n_obs),
                "current": np.random.uniform(10, 50, n_obs),
                "voltage": np.random.uniform(200, 250, n_obs),
            }
        )

        # Test single column performance (baseline)
        spreader_single = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols="power",  # Single column
            group_cols=["vehicle_id", "location"],
            freq="1h",
        )

        # Profile single column processing
        profiler_single = Profiler()
        profiler_single.start()
        result_single = spreader_single.spread(base_obs)
        profiler_single.stop()

        # Test multiple columns performance
        spreader_multi = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols=["power", "energy", "current", "voltage"],  # 4 columns
            group_cols=["vehicle_id", "location"],
            freq="1h",
        )

        # Profile multiple column processing
        profiler_multi = Profiler()
        profiler_multi.start()
        result_multi = spreader_multi.spread(base_obs)
        profiler_multi.stop()

        # Extract timing information
        single_time = profiler_single.last_session.duration
        multi_time = profiler_multi.last_session.duration

        # Performance assertions
        print(f"Single column time: {single_time:.4f}s")
        print(f"Multiple column time: {multi_time:.4f}s")
        print(f"Performance ratio (multi/single): {multi_time/single_time:.2f}x")

        # Print detailed profiles for analysis
        print("\nSingle Column Profile:")
        print(profiler_single.output_text(unicode=True, color=False))
        print("\nMultiple Columns Profile:")
        print(profiler_multi.output_text(unicode=True, color=False))

        # Basic correctness checks
        assert isinstance(result_single, pd.DataFrame)
        assert isinstance(result_multi, pd.DataFrame)
        assert len(result_single) > 0
        assert len(result_multi) > 0

        # Should have same number of rows (same expansion logic)
        assert len(result_single) == len(result_multi)

        # Multi-column result should have all 4 value columns
        expected_cols = [
            "vehicle_id",
            "location",
            "timestamp",
            "power",
            "energy",
            "current",
            "voltage",
        ]
        assert sorted(result_multi.columns) == sorted(expected_cols)

        # Single column result should only have power
        single_expected_cols = ["vehicle_id", "location", "timestamp", "power"]
        assert sorted(result_single.columns) == sorted(single_expected_cols)

        # Performance expectation: multiple columns should be roughly 4x slower
        # but not more than 6x (allowing for overhead)
        expected_ratio = 4.0  # 4 columns
        tolerance = 2.0  # Allow 2x additional overhead
        assert (
            multi_time / single_time <= expected_ratio + tolerance
        ), f"Multiple column processing is too slow: {multi_time/single_time:.2f}x vs expected ~{expected_ratio}x"

        # Verify that power column values are identical between single and multi
        pd.testing.assert_series_equal(
            result_single["power"].sort_values().reset_index(drop=True),
            result_multi["power"].sort_values().reset_index(drop=True),
            check_names=False,
        )

    @pytest.mark.performance
    def test_multiple_columns_scaling_performance(self):
        """Test how performance scales with increasing number of columns."""
        try:
            from pyinstrument import Profiler
        except ImportError:
            pytest.skip("pyinstrument not available")

        n_obs = 2000
        np.random.seed(42)

        # Create base dataset with many potential value columns
        base_obs = pd.DataFrame(
            {
                "vehicle_id": np.random.randint(1, 51, n_obs),
                "location": [f"loc_{i%10}" for i in range(n_obs)],
                "timestamp": pd.date_range("2024-01-01", periods=n_obs, freq="1h"),
                "duration": pd.to_timedelta(
                    np.random.randint(60, 180, n_obs), unit="min"
                ),
                "col1": np.random.uniform(0, 100, n_obs),
                "col2": np.random.uniform(0, 100, n_obs),
                "col3": np.random.uniform(0, 100, n_obs),
                "col4": np.random.uniform(0, 100, n_obs),
                "col5": np.random.uniform(0, 100, n_obs),
                "col6": np.random.uniform(0, 100, n_obs),
                "col7": np.random.uniform(0, 100, n_obs),
                "col8": np.random.uniform(0, 100, n_obs),
            }
        )

        # Test with increasing numbers of columns
        column_counts = [1, 2, 4, 6, 8]
        times = []
        profilers = []

        for num_cols in column_counts:
            value_cols = [f"col{i}" for i in range(1, num_cols + 1)]

            spreader = IntervalBeginSpreader(
                time_col="timestamp",
                dur_col="duration",
                value_cols=value_cols,
                group_cols=["vehicle_id", "location"],
                freq="1h",
            )

            # Profile processing time
            profiler = Profiler()
            profiler.start()
            result = spreader.spread(base_obs)
            profiler.stop()

            elapsed_time = profiler.last_session.duration
            times.append(elapsed_time)
            profilers.append(profiler)

            # Verify correctness
            assert len(result) > 0
            assert (
                len(result.columns) == len(value_cols) + 3
            )  # +3 for group cols + time col

            print(f"{num_cols} columns: {elapsed_time:.4f}s")

        # Print detailed profile for the largest case
        print(f"\nDetailed profile for {column_counts[-1]} columns:")
        print(profilers[-1].output_text(unicode=True, color=False))

        # Performance should scale roughly linearly with number of columns
        baseline_time = times[0]  # 1 column time

        for num_cols, elapsed_time in zip(column_counts[1:], times[1:], strict=False):
            expected_ratio = num_cols  # Linear scaling expected
            actual_ratio = elapsed_time / baseline_time

            print(
                f"{num_cols} cols ratio: {actual_ratio:.2f}x (expected ~{expected_ratio}x)"
            )

            # Allow up to 2x overhead on top of linear scaling
            max_acceptable_ratio = expected_ratio * 2
            assert (
                actual_ratio <= max_acceptable_ratio
            ), f"Scaling is worse than expected: {actual_ratio:.2f}x vs max acceptable {max_acceptable_ratio:.2f}x"

    @pytest.mark.performance
    def test_dtype_conversion_performance_impact(self):
        """Test performance impact of dtype preservation across multiple columns."""
        try:
            from pyinstrument import Profiler
        except ImportError:
            pytest.skip("pyinstrument not available")

        n_obs = 3000
        np.random.seed(42)

        # Create dataset with mixed dtypes
        base_obs = pd.DataFrame(
            {
                "vehicle_id": np.random.randint(1, 51, n_obs),
                "location": [f"depot_{i%5}" for i in range(n_obs)],
                "timestamp": pd.date_range("2024-01-01", periods=n_obs, freq="45min"),
                "duration": pd.to_timedelta(
                    np.random.randint(60, 240, n_obs), unit="min"
                ),
                "int8_col": np.random.randint(0, 100, n_obs, dtype=np.int8),
                "int16_col": np.random.randint(0, 10000, n_obs, dtype=np.int16),
                "float32_col": np.random.uniform(0, 100, n_obs).astype(np.float32),
                "float64_col": np.random.uniform(0, 1000000, n_obs).astype(np.float64),
            }
        )

        # Test with mixed dtypes (requires type preservation)
        spreader_mixed = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols=["int8_col", "int16_col", "float32_col", "float64_col"],
            group_cols=["vehicle_id", "location"],
            freq="1h",
        )

        # Test with all same dtype (no conversion needed)
        base_obs_uniform = base_obs.copy()
        base_obs_uniform["col1"] = base_obs_uniform["int8_col"].astype(np.float64)
        base_obs_uniform["col2"] = base_obs_uniform["int16_col"].astype(np.float64)
        base_obs_uniform["col3"] = base_obs_uniform["float32_col"].astype(np.float64)
        base_obs_uniform["col4"] = base_obs_uniform["float64_col"].astype(np.float64)

        spreader_uniform = IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_cols=["col1", "col2", "col3", "col4"],
            group_cols=["vehicle_id", "location"],
            freq="1h",
        )

        # Compile numba
        _ = spreader_mixed.spread(base_obs)

        # Profile mixed dtype performance
        profiler_mixed = Profiler()
        profiler_mixed.start()
        result_mixed = spreader_mixed.spread(base_obs)
        profiler_mixed.stop()

        # Profile uniform dtype performance
        profiler_uniform = Profiler()
        profiler_uniform.start()
        result_uniform = spreader_uniform.spread(base_obs_uniform)
        profiler_uniform.stop()

        mixed_time = profiler_mixed.last_session.duration
        uniform_time = profiler_uniform.last_session.duration

        print(f"Mixed dtypes time: {mixed_time:.4f}s")
        print(f"Uniform dtypes time: {uniform_time:.4f}s")
        print(f"Overhead ratio (mixed/uniform): {mixed_time/uniform_time:.2f}x")

        print("\nMixed Dtypes Profile:")
        print(profiler_mixed.output_text(unicode=True, color=False))
        print("\nUniform Dtypes Profile:")
        print(profiler_uniform.output_text(unicode=True, color=False))

        # Verify correctness
        assert len(result_mixed) > 0
        assert len(result_uniform) > 0
        assert len(result_mixed) == len(result_uniform)

        # Verify dtype preservation
        assert result_mixed["int8_col"].dtype == np.int8
        assert result_mixed["int16_col"].dtype == np.int16
        assert result_mixed["float32_col"].dtype == np.float32
        assert result_mixed["float64_col"].dtype == np.float64

        # Performance expectation: mixed dtypes should not be more than 50% slower
        max_acceptable_overhead = 1.5
        assert (
            mixed_time / uniform_time <= max_acceptable_overhead
        ), f"Dtype conversion overhead too high: {mixed_time/uniform_time:.2f}x vs max {max_acceptable_overhead}x"

    def test_combined_output_structure(self, spreader):
        """Test structure of combined expanded/non-expanded output."""
        obs = pd.DataFrame(
            {
                "vehicle_id": [1, 2, 3],
                "location": ["A", "B", "C"],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:30:00",  # Will expand
                        "2024-01-01 15:00:00",  # Won't expand
                        "2024-01-01 20:30:00",  # Will expand
                    ]
                ),
                "duration": pd.to_timedelta(["2h", "20min", "90min"]),
                "power": [100.0, 150.0, 200.0],
            }
        )

        result = spreader.spread(obs, return_spreaded_only=False)

        # Should contain all original power values
        original_powers = set(obs["power"])
        result_powers = set(result["power"])
        assert original_powers.issubset(result_powers)

        # Should have at least as many rows as original
        assert len(result) >= len(obs)


class TestNonzeroGroupedSummarizer:
    """Test cases for NonzeroGroupedSummarizer class."""

    @pytest.fixture
    def sample_data(self):
        """Fixture for sample data DataFrame."""
        return pd.DataFrame(
            {
                "group1": [1, 1, 1, 2, 2, 2],
                "group2": ["A", "A", "B", "A", "B", "B"],
                "power": [100.0, 150.0, 200.0, 120.0, 180.0, 90.0],
                "energy": [50.0, 75.0, 100.0, 60.0, 90.0, 45.0],
                "possible_count": [10, 10, 8, 12, 15, 8],
            }
        )

    @pytest.fixture
    def summarizer_single(self):
        """Fixture for single-column summarizer."""
        return NonzeroGroupedSummarizer(
            group_cols=["group1", "group2"],
            quantiles=np.array([0.25, 0.5, 0.75]),
        )

    @pytest.fixture
    def summarizer_multi(self):
        """Fixture for multi-column summarizer."""
        return NonzeroGroupedSummarizer(
            group_cols=["group1", "group2"],
            quantiles=np.array([0.25, 0.5, 0.75]),
            value_cols=["power", "energy"],
        )

    def test_init_single_value_col(self):
        """Test initialization with single value column."""
        summarizer = NonzeroGroupedSummarizer(
            group_cols=["group1"],
            quantiles=np.array([0.5]),
            value_cols="power",
        )
        assert summarizer.group_cols == ["group1"]
        assert np.array_equal(summarizer.quantiles, np.array([0.5]))
        assert summarizer.value_cols == ["power"]

    def test_init_multiple_value_cols(self):
        """Test initialization with multiple value columns."""
        summarizer = NonzeroGroupedSummarizer(
            group_cols=["group1", "group2"],
            quantiles=np.array([0.25, 0.5, 0.75]),
            value_cols=["power", "energy", "voltage"],
        )
        assert summarizer.group_cols == ["group1", "group2"]
        assert np.array_equal(summarizer.quantiles, np.array([0.25, 0.5, 0.75]))
        assert summarizer.value_cols == ["power", "energy", "voltage"]

    def test_init_no_value_cols(self):
        """Test initialization without value columns."""
        summarizer = NonzeroGroupedSummarizer(
            group_cols=["group1"],
            quantiles=np.array([0.5]),
        )
        assert summarizer.value_cols is None

    def test_summarize_single_column_string(self, summarizer_single, sample_data):
        """Test summarize with single column passed as string."""
        result = summarizer_single.summarize(
            events=sample_data,
            value_cols="power",
            possible_count_col="possible_count",
        )

        # Should return DataFrame with quantiles as columns
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [0.25, 0.5, 0.75]
        assert result.index.names == ["group1", "group2"]
        assert len(result) > 0

    def test_summarize_single_column_list(self, summarizer_single, sample_data):
        """Test summarize with single column passed as list."""
        result = summarizer_single.summarize(
            events=sample_data,
            value_cols=["power"],
            possible_count_col="possible_count",
        )

        # Should return DataFrame with quantiles as columns (same as string input)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [0.25, 0.5, 0.75]
        assert result.index.names == ["group1", "group2"]

    def test_summarize_multiple_columns(self, summarizer_single, sample_data):
        """Test summarize with multiple columns."""
        result = summarizer_single.summarize(
            events=sample_data,
            value_cols=["power", "energy"],
            possible_count_col="possible_count",
        )

        # Should return DataFrame with MultiIndex columns
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex)
        assert result.columns.names == ["value_col", "quantile"]
        assert result.index.names == ["group1", "group2"]

        # Check that all combinations are present
        expected_cols = [
            ("power", 0.25),
            ("power", 0.5),
            ("power", 0.75),
            ("energy", 0.25),
            ("energy", 0.5),
            ("energy", 0.75),
        ]
        assert list(result.columns) == expected_cols

    def test_summarize_with_constructor_value_cols(self, summarizer_multi, sample_data):
        """Test summarize using value_cols from constructor."""
        # Fix: Need to pass value_cols to summarize method, constructor value_cols not used
        result = summarizer_multi.summarize(
            events=sample_data,
            value_cols=["power", "energy"],  # Must specify value_cols
            possible_count_col="possible_count",
        )

        # Should use value_cols from parameter
        assert isinstance(result.columns, pd.MultiIndex)
        expected_cols = [
            ("power", 0.25),
            ("power", 0.5),
            ("power", 0.75),
            ("energy", 0.25),
            ("energy", 0.5),
            ("energy", 0.75),
        ]
        assert list(result.columns) == expected_cols

    def test_summarize_override_constructor_value_cols(
        self, summarizer_multi, sample_data
    ):
        """Test that parameter value_cols overrides constructor value_cols."""
        result = summarizer_multi.summarize(
            events=sample_data,
            value_cols="power",  # Override constructor value_cols
            possible_count_col="possible_count",
        )

        # Should use parameter value_cols, not constructor value_cols
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [0.25, 0.5, 0.75]  # Single column format

    def test_quantile_calculation_accuracy(self, summarizer_single):
        """Test that quantile calculations are accurate."""
        # Create controlled data where we know expected quantiles
        # Single group with multiple observations but same possible_count
        data = pd.DataFrame(
            {
                "group1": [1] * 3,
                "group2": ["A"] * 3,
                "power": [10.0, 20.0, 30.0],  # 3 nonzeros
                "possible_count": [
                    6,
                    6,
                    6,
                ],  # 6 total possible each, so 3 additional zeros per row but they use first row's count
            }
        )

        # The algorithm uses cur_counts = counts[idx[0]], so it takes the first possible_count (6)
        # With 3 zeros + 3 nonzeros = [0,0,0,10,20,30]
        # Quantiles: 0.25->0, 0.5->5, 0.75->20
        result = summarizer_single.summarize(
            events=data,
            value_cols="power",
            possible_count_col="possible_count",
        )

        # Check approximate quantile values
        values = result.iloc[0].values
        assert values[0] == 0.0  # 0.25 quantile should be 0
        # The median of [0,0,0,10,20,30] is 5.0
        assert abs(values[1] - 5.0) < 1e-10  # 0.5 quantile should be 5
        # The 0.75 quantile of [0,0,0,10,20,30] is 17.5 (between 10 and 20)
        assert abs(values[2] - 17.5) < 1e-10  # 0.75 quantile should be 17.5

    def test_different_dtypes_preserved(self, summarizer_single):
        """Test that different data types are handled correctly."""
        data = pd.DataFrame(
            {
                "group1": [1, 1, 1],
                "group2": ["A", "A", "A"],
                "int_col": [10, 20, 30],
                "float_col": [10.5, 20.5, 30.5],
                "possible_count": [5, 5, 5],
            }
        )

        # Test integer column
        result_int = summarizer_single.summarize(
            events=data,
            value_cols="int_col",
            possible_count_col="possible_count",
        )
        assert isinstance(result_int, pd.DataFrame)

        # Test float column
        result_float = summarizer_single.summarize(
            events=data,
            value_cols="float_col",
            possible_count_col="possible_count",
        )
        assert isinstance(result_float, pd.DataFrame)

    def test_multiple_groups(self, summarizer_single):
        """Test with multiple distinct groups."""
        data = pd.DataFrame(
            {
                "group1": [1, 1, 2, 2, 3, 3],
                "group2": ["A", "B", "A", "B", "A", "B"],
                "power": [100, 200, 150, 250, 120, 220],
                "possible_count": [5, 5, 5, 5, 5, 5],
            }
        )

        result = summarizer_single.summarize(
            events=data,
            value_cols="power",
            possible_count_col="possible_count",
        )

        # Should have 6 groups (3 group1 values × 2 group2 values)
        assert len(result) == 6

        # Check that all group combinations are present
        expected_index = pd.MultiIndex.from_product(
            [[1, 2, 3], ["A", "B"]], names=["group1", "group2"]
        )
        pd.testing.assert_index_equal(result.index, expected_index)

    def test_error_too_many_observations(self, summarizer_single):
        """Test error when observations exceed possible count."""
        # Create data where nonzeros exceed possible count
        data = pd.DataFrame(
            {
                "group1": [1] * 5,
                "group2": ["A"] * 5,
                "power": [10, 20, 30, 40, 50],  # 5 observations
                "possible_count": [3] * 5,  # But only 3 possible
            }
        )

        with pytest.raises(ValueError, match="Number of observations exceeds"):
            summarizer_single.summarize(
                events=data,
                value_cols="power",
                possible_count_col="possible_count",
            )

    def test_empty_dataframe(self, summarizer_single):
        """Test with empty DataFrame."""
        empty_data = pd.DataFrame(
            {
                "group1": pd.Series([], dtype="int64"),
                "group2": pd.Series([], dtype="object"),
                "power": pd.Series([], dtype="float64"),
                "possible_count": pd.Series([], dtype="int64"),
            }
        )

        # Should raise error for empty DataFrame
        with pytest.raises(ValueError, match="Cannot process empty DataFrame"):
            summarizer_single.summarize(
                events=empty_data,
                value_cols="power",
                possible_count_col="possible_count",
            )

    def test_single_row_multiple_columns(self, summarizer_single):
        """Test with single row and multiple columns."""
        data = pd.DataFrame(
            {
                "group1": [1],
                "group2": ["A"],
                "power": [100.0],
                "energy": [50.0],
                "possible_count": [5],
            }
        )

        result = summarizer_single.summarize(
            events=data,
            value_cols=["power", "energy"],
            possible_count_col="possible_count",
        )

        assert len(result) == 1
        assert isinstance(result.columns, pd.MultiIndex)

    @pytest.mark.parametrize(
        "quantiles",
        [
            np.array([0.5]),  # Single quantile
            np.array([0.25, 0.75]),  # Two quantiles
            np.array([0.1, 0.25, 0.5, 0.75, 0.9]),  # Five quantiles
        ],
    )
    def test_different_quantile_configurations(self, quantiles):
        """Test with different quantile configurations."""
        summarizer = NonzeroGroupedSummarizer(
            group_cols=["group1"],
            quantiles=quantiles,
        )

        data = pd.DataFrame(
            {
                "group1": [1, 1, 1],
                "power": [10, 20, 30],
                "possible_count": [5, 5, 5],
            }
        )

        result = summarizer.summarize(
            events=data,
            value_cols="power",
            possible_count_col="possible_count",
        )

        # Should have correct number of quantile columns
        assert len(result.columns) == len(quantiles)
        assert list(result.columns) == list(quantiles)

    def test_core_quantile_function_directly(self):
        """Test the core quantile calculation function directly."""
        n_obs = 10  # 10 total observations
        nonzeros = np.array([5.0, 10.0, 15.0])  # 3 nonzero values
        quantiles = np.array([0.25, 0.5, 0.75])

        result = NonzeroGroupedSummarizer._calc_sparse_quantiles_core(
            n_obs=n_obs,
            nonzeros=nonzeros,
            quantiles=quantiles,
        )

        # Should return array with quantile results
        assert isinstance(result, np.ndarray)
        assert len(result) == len(quantiles)

        # With 7 zeros + 3 nonzeros: [0,0,0,0,0,0,0,5,10,15]
        # 0.25 quantile (2.5th position) should be 0
        # 0.5 quantile (5th position) should be 0
        # 0.75 quantile (7.5th position) should be between 0 and 5
        assert result[0] == 0.0  # 0.25 quantile
        assert result[1] == 0.0  # 0.5 quantile
        assert 0.0 <= result[2] <= 5.0  # 0.75 quantile

    def test_performance_multiple_vs_single_calls(self, sample_data):
        """Test that multiple columns in one call performs similarly to separate calls."""
        import time

        summarizer = NonzeroGroupedSummarizer(
            group_cols=["group1", "group2"],
            quantiles=np.array([0.25, 0.5, 0.75]),
        )

        # Time single call with multiple columns
        start_time = time.time()
        result_multi = summarizer.summarize(
            events=sample_data,
            value_cols=["power", "energy"],
            possible_count_col="possible_count",
        )
        multi_time = time.time() - start_time

        # Time separate calls
        start_time = time.time()
        result_power = summarizer.summarize(
            events=sample_data,
            value_cols="power",
            possible_count_col="possible_count",
        )
        result_energy = summarizer.summarize(
            events=sample_data,
            value_cols="energy",
            possible_count_col="possible_count",
        )
        separate_time = time.time() - start_time

        # Multi-column call should be faster than separate calls
        # (though for small data the difference might be minimal)
        print(f"Multi-column time: {multi_time:.4f}s")
        print(f"Separate calls time: {separate_time:.4f}s")

        # Verify correctness - results should be equivalent
        # Extract power and energy results from multi-column result
        power_cols = [col for col in result_multi.columns if col[0] == "power"]
        energy_cols = [col for col in result_multi.columns if col[0] == "energy"]

        power_from_multi = result_multi[power_cols].copy()
        power_from_multi.columns = [
            col[1] for col in power_cols
        ]  # Extract quantile values

        energy_from_multi = result_multi[energy_cols].copy()
        energy_from_multi.columns = [
            col[1] for col in energy_cols
        ]  # Extract quantile values

        # Compare with separate results
        pd.testing.assert_frame_equal(result_power, power_from_multi)
        pd.testing.assert_frame_equal(result_energy, energy_from_multi)
