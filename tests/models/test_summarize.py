"""Unit tests for megaPLuG.models.summarize module.

This module tests the EventExpander and NonzeroGroupedSummarizer classes using pytest.
"""

import numpy as np
import pandas as pd
import pytest
from megaPLuG.models.summarize import EventExpander


class TestEventExpander:
    """Test cases for EventExpander class."""

    @pytest.fixture
    def expander(self):
        """Fixture for EventExpander instance."""
        return EventExpander(
            time_col="timestamp",
            dur_col="duration",
            value_col="power",
            group_cols=["vehicle_id", "location"],
            freq="1h",
        )

    @pytest.fixture
    def sample_events(self):
        """Fixture for sample events DataFrame."""
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
        expander = EventExpander(
            time_col="time",
            dur_col="dur",
            value_col="val",
            group_cols=["group1", "group2"],
            freq="15min",
        )

        assert expander.time_col == "time"
        assert expander.dur_col == "dur"
        assert expander.value_col == "val"
        assert expander.group_cols == ["group1", "group2"]
        assert expander.freq == "15min"

    def test_init_empty_group_cols(self):
        """Test initialization with empty group_cols list."""
        expander = EventExpander(
            time_col="time", dur_col="dur", value_col="val", group_cols=[], freq="1h"
        )

        assert expander.group_cols == []

    def test_expand_events_no_expansion_needed(self, expander, sample_events):
        """Test events that don't cross time boundaries."""
        result = expander.expand_events(sample_events)

        # Should return original events since no expansion is needed
        expected_cols = expander.group_cols + [expander.time_col, expander.value_col]
        assert list(result.columns) == expected_cols
        assert len(result) == len(sample_events)

    def test_expand_events_single_expansion(self, expander):
        """Test event spanning exactly 2 time periods."""
        # Events with different expansion needs
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        # Should have more rows due to expansion
        assert len(result) >= len(events)

        # Check that power value is preserved
        assert all(result["power"] == 100.0)

    def test_expand_events_multiple_expansions(self, expander):
        """Test event spanning multiple time periods."""
        # Events with different expansion needs
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        # Should have expanded rows
        expanded_rows = result[result["power"] == 200.0]
        assert len(expanded_rows) > 1

    def test_expand_events_different_frequencies(self):
        """Test with different frequency settings."""
        events = pd.DataFrame(
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
        expander_15min = EventExpander(
            time_col="timestamp",
            dur_col="duration",
            value_col="power",
            group_cols=["vehicle_id", "location"],
            freq="15min",
        )

        result = expander_15min.expand_events(events)

        # Should have more expanded rows with finer granularity
        expanded_rows = result[result["power"] == 100.0]
        assert len(expanded_rows) > 1

    def test_timezone_naive_input(self, expander):
        """Test DataFrame with timezone-naive timestamps."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        # Should process without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_timezone_utc_input(self, expander):
        """Test DataFrame with UTC timestamps."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        # Should process without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_timezone_other_raises_error(self, expander):
        """Test that non-UTC/non-naive timezones raise RuntimeError."""
        events = pd.DataFrame(
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
            expander.expand_events(events)

    def test_return_expansions_only_true(self, expander):
        """Test return_expansions_only=True."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events, return_expansions_only=True)

        # Should only contain expanded events
        assert len(result) > 0
        # All returned rows should be from expanded events
        assert all(result["power"] == 100.0)  # Only first event gets expanded

    def test_return_expansions_only_false(self, expander):
        """Test return_expansions_only=False (default)."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events, return_expansions_only=False)

        # Should contain both expanded and non-expanded events
        assert len(result) >= len(events)
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
    def test_different_value_types(self, expander, value_type, test_value):
        """Test with different value types."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_multiple_group_columns(self):
        """Test with multiple group columns."""
        events = pd.DataFrame(
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

        expander_multi = EventExpander(
            time_col="timestamp",
            dur_col="duration",
            value_col="power",
            group_cols=["vehicle_id", "location", "driver"],
            freq="1h",
        )

        result = expander_multi.expand_events(events)

        # Should preserve all group columns
        expected_cols = ["vehicle_id", "location", "driver", "timestamp", "power"]
        assert sorted(result.columns) == sorted(expected_cols)

    def test_single_group_column(self):
        """Test with single group column."""
        events = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 11:30:00"]
                ),
                "duration": pd.to_timedelta(["90min", "90min"]),
                "power": [100.0, 150.0],
            }
        )

        expander_single = EventExpander(
            time_col="timestamp",
            dur_col="duration",
            value_col="power",
            group_cols=["vehicle_id"],
            freq="1h",
        )

        result = expander_single.expand_events(events)

        # Should preserve single group column
        expected_cols = ["vehicle_id", "timestamp", "power"]
        assert sorted(result.columns) == sorted(expected_cols)

    def test_zero_duration_events(self, expander):
        """Test events with zero duration."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        # Should handle zero duration gracefully
        assert isinstance(result, pd.DataFrame)
        expected_cols = expander.group_cols + [expander.time_col, expander.value_col]
        assert list(result.columns) == expected_cols

    def test_negative_duration_events(self, expander):
        """Test events with negative duration."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        # Should handle negative duration gracefully
        assert isinstance(result, pd.DataFrame)
        expected_cols = expander.group_cols + [expander.time_col, expander.value_col]
        assert list(result.columns) == expected_cols

    def test_empty_dataframe(self, expander):
        """Test empty input DataFrame."""
        events = pd.DataFrame(
            columns=["vehicle_id", "location", "timestamp", "duration", "power"]
        )
        events["timestamp"] = pd.to_datetime(events["timestamp"])
        events["duration"] = pd.to_timedelta(events["duration"])

        result = expander.expand_events(events)

        # Should return empty DataFrame with correct columns
        expected_cols = expander.group_cols + [expander.time_col, expander.value_col]
        assert list(result.columns) == expected_cols
        assert len(result) == 0

    def test_minimal_dataframe(self, expander):
        """Test DataFrame with minimal events."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_events_at_time_boundaries(self, expander):
        """Test events starting/ending exactly on frequency boundaries."""
        # Events starting exactly at hour boundaries
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_core_algorithm_single_event(self):
        """Test the core numba function directly."""
        starts = np.array([0], dtype=np.int64)
        ends = np.array([3600000000000], dtype=np.int64)  # 1 hour in nanoseconds
        vals = np.array([100.0])
        grps = np.array([1])
        tstep_ns = 3600000000000  # 1 hour in nanoseconds

        grps_exp, times_exp, vals_exp = EventExpander._expand_events_core(
            starts, ends, vals, grps, tstep_ns
        )

        assert isinstance(grps_exp, np.ndarray)
        assert isinstance(times_exp, np.ndarray)
        assert isinstance(vals_exp, np.ndarray)
        assert len(grps_exp) == len(times_exp)
        assert len(times_exp) == len(vals_exp)

    def test_core_multiple_events(self):
        """Test core function with multiple events."""
        starts = np.array([0, 3600000000000], dtype=np.int64)
        ends = np.array(
            [7200000000000, 10800000000000], dtype=np.int64
        )  # 2 and 3 hours
        vals = np.array([100.0, 200.0])
        grps = np.array([1, 2])
        tstep_ns = 3600000000000  # 1 hour in nanoseconds

        grps_exp, times_exp, vals_exp = EventExpander._expand_events_core(
            starts, ends, vals, grps, tstep_ns
        )

        # Should have expanded multiple events
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

        grps_exp, times_exp, vals_exp = EventExpander._expand_events_core(
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

        grps_exp, times_exp, vals_exp = EventExpander._expand_events_core(
            starts, ends, vals, grps, tstep_ns
        )

        assert isinstance(vals_exp, np.ndarray)
        assert vals_exp.dtype == dtype

    def test_wrapper_function(self, expander, mocker):
        """Test the wrapper function with mocked dependencies."""
        # Mock the utility functions
        mock_get_dtype = mocker.patch("megaPLuG.models.summarize.get_basic_dtype_ser")
        mock_get_dtype.side_effect = lambda x: x  # Return the series as-is

        # Create test data that would normally require expansion
        events = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:30:00", "2024-01-01 14:00:00"], utc=True
                ),
                "duration": pd.to_timedelta(["90min", "75min"]),
                "power": [100.0, 110.0],
                "codes": [0, 1],  # Required by wrapper
            }
        )

        result_df = expander._expand_events_wrapper(events)

        # Should return a DataFrame
        assert isinstance(result_df, pd.DataFrame)
        assert "codes" in result_df.columns
        assert expander.time_col in result_df.columns
        assert expander.value_col in result_df.columns

    def test_missing_required_columns(self, expander):
        """Test behavior with missing required columns."""
        # Missing duration column
        events = pd.DataFrame(
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
            expander.expand_events(events)

    def test_indexed_dataframe_input(self, expander):
        """Test with pre-indexed DataFrames."""
        events = pd.DataFrame(
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
        events = events.set_index(["vehicle_id", "location"])

        result = expander.expand_events(events)

        # Should handle indexed input gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_output_column_preservation(self, expander):
        """Test that output contains correct columns."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        expected_cols = expander.group_cols + [expander.time_col, expander.value_col]
        assert sorted(result.columns) == sorted(expected_cols)

    def test_output_row_count_logic(self, expander):
        """Test expected number of rows in output."""
        # Events that should expand to multiple time periods
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        # Should have at least the original row, likely more due to expansion
        assert len(result) >= 1

    def test_value_propagation(self, expander):
        """Test that values are correctly propagated to expanded timestamps."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        # All power values should be 100.0 for this vehicle/location
        vehicle_data = result[(result["vehicle_id"] == 1) & (result["location"] == "A")]
        assert all(vehicle_data["power"] == 100.0)

    def test_group_preservation(self, expander):
        """Test that group values are maintained correctly."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        # Each expanded row should maintain its group values
        vehicle_1_data = result[result["vehicle_id"] == 1]
        vehicle_2_data = result[result["vehicle_id"] == 2]

        # Vehicle 1 should only have location A and power 100.0
        assert all(vehicle_1_data["location"] == "A")
        assert all(vehicle_1_data["power"] == 100.0)

        # Vehicle 2 should only have location B and power 200.0
        assert all(vehicle_2_data["location"] == "B")
        assert all(vehicle_2_data["power"] == 200.0)

    def test_real_world_scenario(self, expander):
        """Test with realistic vehicle telematics data."""
        # Simulate realistic charging events
        events = pd.DataFrame(
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

        result = expander.expand_events(events)

        # Should handle realistic scenario without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= len(events)

        # Check that all original power values are preserved
        original_powers = set(events["power"])
        result_powers = set(result["power"])
        assert original_powers.issubset(result_powers)

    @pytest.mark.performance
    def test_large_dataset_performance(self, expander):
        """Test performance with large datasets."""
        # Create a larger dataset
        n_events = 1000
        events = pd.DataFrame(
            {
                "vehicle_id": np.random.randint(1, 101, n_events),
                "location": [f"location_{i%10}" for i in range(n_events)],
                "timestamp": pd.date_range("2024-01-01", periods=n_events, freq="1h"),
                "duration": pd.to_timedelta(
                    np.random.randint(30, 180, n_events), unit="min"
                ),
                "power": np.random.uniform(50, 200, n_events),
            }
        )

        # Should complete without errors (performance is measured externally)
        result = expander.expand_events(events)

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= len(events)

    def test_timezone_conversion_consistency(self, expander):
        """Test that output timezone matches input timezone format."""
        # Test with timezone-naive input
        events_naive = pd.DataFrame(
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

        result_naive = expander.expand_events(events_naive)

        # Output should also be timezone-naive
        assert result_naive["timestamp"].dt.tz is None

        # Test with UTC input
        events_utc = pd.DataFrame(
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

        result_utc = expander.expand_events(events_utc)

        # Output should maintain UTC timezone
        assert result_utc["timestamp"].dt.tz is not None

    def test_combined_output_structure(self, expander):
        """Test structure of combined expanded/non-expanded output."""
        events = pd.DataFrame(
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

        result = expander.expand_events(events, return_expansions_only=False)

        # Should contain all original power values
        original_powers = set(events["power"])
        result_powers = set(result["power"])
        assert original_powers.issubset(result_powers)

        # Should have at least as many rows as original
        assert len(result) >= len(events)
