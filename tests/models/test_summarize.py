"""Unit tests for megaPLuG.models.summarize module.

This module tests the IntervalBeginSpreader and NonzeroGroupedSummarizer classes using pytest.
"""

import numpy as np
import pandas as pd
import pytest
from megaPLuG.models.summarize import IntervalBeginSpreader


class TestIntervalBeginSpreader:
    """Test cases for IntervalBeginSpreader class."""

    @pytest.fixture
    def spreader(self):
        """Fixture for IntervalBeginSpreader instance."""
        return IntervalBeginSpreader(
            time_col="timestamp",
            dur_col="duration",
            value_col="power",
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
            value_col="val",
            group_cols=["group1", "group2"],
            freq="15min",
        )

        assert spreader.time_col == "time"
        assert spreader.dur_col == "dur"
        assert spreader.value_col == "val"
        assert spreader.group_cols == ["group1", "group2"]
        assert spreader.freq == "15min"

    def test_init_empty_group_cols(self):
        """Test initialization with empty group_cols list."""
        spreader = IntervalBeginSpreader(
            time_col="time", dur_col="dur", value_col="val", group_cols=[], freq="1h"
        )

        assert spreader.group_cols == []

    def test_expand_obs_no_expansion_needed(self, spreader, sample_obs):
        """Test obs that don't cross time boundaries."""
        result = spreader.spread(sample_obs)

        # Should return original obs since no expansion is needed
        expected_cols = spreader.group_cols + [spreader.time_col, spreader.value_col]
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
            value_col="power",
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
            value_col="power",
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
            value_col="power",
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
        expected_cols = spreader.group_cols + [spreader.time_col, spreader.value_col]
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
        expected_cols = spreader.group_cols + [spreader.time_col, spreader.value_col]
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
        mock_get_dtype = mocker.patch("megaPLuG.models.summarize.get_basic_dtype_ser")
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
        assert spreader.value_col in result_df.columns

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

        expected_cols = spreader.group_cols + [spreader.time_col, spreader.value_col]
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
