"""Comprehensive test suite for megaPLuG.utils.data module.

This module tests the generate_mock_data function using pytest, covering all
data types, edge cases, and integration scenarios.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from megaPLuG.utils.data import generate_mock_data

try:
    import dask.dataframe as dd

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


class TestGenerateMockDataBasic:
    """Test basic functionality of generate_mock_data function."""

    @pytest.fixture
    def simple_meta(self):
        """Simple meta DataFrame with common dtypes."""
        return pd.DataFrame(
            {
                "int_col": pd.Series([], dtype="int64"),
                "float_col": pd.Series([], dtype="float64"),
                "bool_col": pd.Series([], dtype="bool"),
                "str_col": pd.Series([], dtype="object"),
            }
        )

    def test_generates_correct_row_count(self, simple_meta):
        """Test that function generates the requested number of rows."""
        for n_rows in [1, 3, 5, 10]:
            result = generate_mock_data(simple_meta, n_rows=n_rows)
            assert len(result) == n_rows

    def test_preserves_column_names(self, simple_meta):
        """Test that all original column names are preserved."""
        result = generate_mock_data(simple_meta, n_rows=3)
        assert list(result.columns) == list(simple_meta.columns)

    def test_preserves_dtypes(self, simple_meta):
        """Test that all dtypes match the input meta DataFrame."""
        result = generate_mock_data(simple_meta, n_rows=3)
        for col in simple_meta.columns:
            assert result[col].dtype == simple_meta[col].dtype

    def test_reproducibility_with_seed(self, simple_meta):
        """Test that same seed produces identical results."""
        result1 = generate_mock_data(simple_meta, n_rows=5, seed=42)
        result2 = generate_mock_data(simple_meta, n_rows=5, seed=42)
        pd.testing.assert_frame_equal(result1, result2)

    def test_different_seeds_produce_different_results(self, simple_meta):
        """Test that different seeds produce different results."""
        result1 = generate_mock_data(simple_meta, n_rows=10, seed=42)
        result2 = generate_mock_data(simple_meta, n_rows=10, seed=123)
        # At least some values should be different
        assert not result1.equals(result2)

    def test_empty_dataframe_handling(self):
        """Test graceful handling of empty input DataFrame."""
        empty_meta = pd.DataFrame()
        with patch("megaPLuG.utils.data.logger.warning") as mock_warning:
            result = generate_mock_data(empty_meta, n_rows=3)
            mock_warning.assert_called_once()
        assert len(result) == 0
        assert len(result.columns) == 0


class TestGenerateMockDataNumericTypes:
    """Test numeric data type handling."""

    @pytest.mark.parametrize(
        "int_dtype",
        ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"],
    )
    def test_integer_types(self, int_dtype):
        """Test standard integer type generation."""
        meta = pd.DataFrame({"col": pd.Series([], dtype=int_dtype)})
        result = generate_mock_data(meta, n_rows=10)

        assert result["col"].dtype == np.dtype(int_dtype)
        assert result["col"].notna().all()  # No NaN values
        assert (result["col"] >= 0).all()  # Non-negative values
        assert (result["col"] <= 100).all()  # Within expected range

    @pytest.mark.parametrize("int_dtype", ["Int8", "Int16", "Int32", "Int64"])
    def test_nullable_integer_types(self, int_dtype):
        """Test nullable integer type generation."""
        meta = pd.DataFrame({"col": pd.Series([], dtype=int_dtype)})
        result = generate_mock_data(
            meta, n_rows=100, seed=42
        )  # Larger sample for NaN testing

        assert result["col"].dtype.name == int_dtype
        # Should have some NaN values (around 10%)
        nan_count = result["col"].isna().sum()
        assert 0 <= nan_count <= 20  # Reasonable range for 10% of 100 rows
        # Non-NaN values should be in expected range
        valid_values = result["col"].dropna()
        if len(valid_values) > 0:
            assert (valid_values >= 0).all()
            assert (valid_values <= 100).all()

    @pytest.mark.parametrize("float_dtype", ["float32", "float64"])
    def test_float_types(self, float_dtype):
        """Test standard float type generation."""
        meta = pd.DataFrame({"col": pd.Series([], dtype=float_dtype)})
        result = generate_mock_data(meta, n_rows=10)

        assert result["col"].dtype == np.dtype(float_dtype)
        assert result["col"].notna().all()  # No NaN values
        assert (result["col"] >= 0.0).all()
        assert (result["col"] <= 100.0).all()

    @pytest.mark.parametrize("float_dtype", ["Float32", "Float64"])
    def test_nullable_float_types(self, float_dtype):
        """Test nullable float type generation."""
        meta = pd.DataFrame({"col": pd.Series([], dtype=float_dtype)})
        result = generate_mock_data(meta, n_rows=100, seed=42)

        assert result["col"].dtype.name == float_dtype
        # Should have some NaN values
        nan_count = result["col"].isna().sum()
        assert 0 <= nan_count <= 20
        # Non-NaN values should be in expected range
        valid_values = result["col"].dropna()
        if len(valid_values) > 0:
            assert (valid_values >= 0.0).all()
            assert (valid_values <= 100.0).all()


class TestGenerateMockDataBooleanTypes:
    """Test boolean data type handling."""

    def test_standard_bool_type(self):
        """Test standard boolean type generation."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="bool")})
        result = generate_mock_data(meta, n_rows=50, seed=42)

        assert result["col"].dtype == np.dtype("bool")
        assert result["col"].notna().all()
        # Should have mix of True and False
        assert result["col"].sum() > 0  # At least one True
        assert (~result["col"]).sum() > 0  # At least one False

    def test_nullable_boolean_type(self):
        """Test nullable boolean type generation."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="boolean")})
        result = generate_mock_data(meta, n_rows=100, seed=42)

        assert result["col"].dtype.name == "boolean"
        # Should have some NaN values
        nan_count = result["col"].isna().sum()
        assert 0 <= nan_count <= 20
        # Non-NaN values should be boolean
        valid_values = result["col"].dropna()
        if len(valid_values) > 0:
            assert valid_values.dtype.name == "boolean"


class TestGenerateMockDataStringTypes:
    """Test string and object data type handling."""

    def test_object_dtype_strings(self):
        """Test object dtype treated as strings."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="object")})
        result = generate_mock_data(meta, n_rows=5)

        assert result["col"].dtype == np.dtype("object")
        assert result["col"].notna().all()
        # All values should be strings starting with 'item_'
        assert all(
            isinstance(val, str) and val.startswith("item_") for val in result["col"]
        )

    def test_string_dtype(self):
        """Test nullable string dtype."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="string")})
        result = generate_mock_data(meta, n_rows=100, seed=42)

        assert result["col"].dtype.name == "string"
        # Should have some NaN values
        nan_count = result["col"].isna().sum()
        assert 0 <= nan_count <= 20
        # Non-NaN values should be strings starting with 'string_'
        valid_values = result["col"].dropna()
        if len(valid_values) > 0:
            assert all(val.startswith("string_") for val in valid_values)


class TestGenerateMockDataDatetimeTypes:
    """Test datetime and timedelta data type handling."""

    def test_datetime64_type(self):
        """Test datetime64[ns] type generation."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="datetime64[ns]")})
        result = generate_mock_data(meta, n_rows=10)

        assert pd.api.types.is_datetime64_any_dtype(result["col"].dtype)
        assert result["col"].notna().all()
        # All dates should be in 2024
        assert all(dt.year == 2024 for dt in result["col"])

    def test_timedelta64_type(self):
        """Test timedelta64[ns] type generation."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="timedelta64[ns]")})
        result = generate_mock_data(meta, n_rows=10)

        assert pd.api.types.is_timedelta64_dtype(result["col"].dtype)
        assert result["col"].notna().all()
        # All values should be positive and reasonable (0.5-24 hours)
        hours = result["col"].dt.total_seconds() / 3600
        assert (hours >= 0.5).all()
        assert (hours <= 24.0).all()

    def test_timezone_aware_datetime(self):
        """Test timezone-aware datetime type generation."""
        import pytz

        # Test with UTC timezone
        utc_dtype = pd.Series([], dtype="datetime64[ns, UTC]").dtype
        meta = pd.DataFrame({"col": pd.Series([], dtype=utc_dtype)})
        result = generate_mock_data(meta, n_rows=5)

        assert pd.api.types.is_datetime64_any_dtype(result["col"].dtype)
        assert result["col"].dt.tz is not None
        assert str(result["col"].dt.tz) == "UTC"
        assert result["col"].notna().all()

        # Test with other timezone
        est_tz = pytz.timezone("US/Eastern")
        est_series = (
            pd.Series([], dtype="datetime64[ns]")
            .dt.tz_localize("UTC")
            .dt.tz_convert(est_tz)
        )
        meta_est = pd.DataFrame({"col": est_series})
        result_est = generate_mock_data(meta_est, n_rows=5)

        assert pd.api.types.is_datetime64_any_dtype(result_est["col"].dtype)
        assert result_est["col"].dt.tz is not None
        assert str(result_est["col"].dt.tz) == "US/Eastern"

    def test_mixed_timezone_columns(self):
        """Test DataFrame with both timezone-naive and timezone-aware columns."""

        utc_dtype = pd.Series([], dtype="datetime64[ns, UTC]").dtype
        naive_dtype = pd.Series([], dtype="datetime64[ns]").dtype

        meta = pd.DataFrame(
            {
                "naive_dt": pd.Series([], dtype=naive_dtype),
                "aware_dt": pd.Series([], dtype=utc_dtype),
            }
        )
        result = generate_mock_data(meta, n_rows=3)

        # Naive column should have no timezone
        assert result["naive_dt"].dt.tz is None
        # Aware column should have UTC timezone
        assert str(result["aware_dt"].dt.tz) == "UTC"

    def test_timezone_conversion_regression(self):
        """Test specific case that was causing 'Cannot use .astype to convert' error."""
        import pytz

        # Recreate the exact scenario from the error message
        tz = pytz.timezone("US/Eastern")
        # This mimics a dwell_end_time column that is timezone-aware
        datetime_series = (
            pd.Series([], dtype="datetime64[ns]")
            .dt.tz_localize("UTC")
            .dt.tz_convert(tz)
        )
        meta = pd.DataFrame({"dwell_end_time": datetime_series})

        # This should not raise the "Cannot use .astype to convert" error
        result = generate_mock_data(meta, n_rows=3, seed=42)

        assert len(result) == 3
        assert pd.api.types.is_datetime64_any_dtype(result["dwell_end_time"].dtype)
        assert str(result["dwell_end_time"].dt.tz) == "US/Eastern"
        assert result["dwell_end_time"].notna().all()


class TestGenerateMockDataCategoricalTypes:
    """Test categorical data type handling."""

    def test_categorical_with_existing_categories(self):
        """Test categorical type with predefined categories."""
        categories = ["A", "B", "C", "D"]
        cat_dtype = pd.CategoricalDtype(categories=categories)
        meta = pd.DataFrame({"col": pd.Series([], dtype=cat_dtype)})
        result = generate_mock_data(meta, n_rows=20, seed=42)

        assert isinstance(result["col"].dtype, pd.CategoricalDtype)
        assert set(result["col"].unique()).issubset(set(categories))

    def test_categorical_with_empty_categories(self):
        """Test categorical type with no predefined categories."""
        cat_dtype = pd.CategoricalDtype(categories=[])
        meta = pd.DataFrame({"col": pd.Series([], dtype=cat_dtype)})
        result = generate_mock_data(meta, n_rows=5)

        assert isinstance(result["col"].dtype, pd.CategoricalDtype)
        # Should create default categories
        assert all(val.startswith("category_") for val in result["col"])


class TestGenerateMockDataIndexHandling:
    """Test index generation and preservation."""

    def test_default_integer_index(self):
        """Test default integer index generation."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="int64")})
        result = generate_mock_data(meta, n_rows=5)

        expected_index = pd.RangeIndex(5)
        pd.testing.assert_index_equal(result.index, expected_index)

    def test_named_integer_index(self):
        """Test named integer index preservation."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="int64")})
        meta.index.name = "my_index"
        result = generate_mock_data(meta, n_rows=5)

        assert result.index.name == "my_index"
        assert pd.api.types.is_integer_dtype(result.index.dtype)

    def test_object_index(self):
        """Test object index generation."""
        meta = pd.DataFrame(
            {"col": pd.Series([], dtype="int64")},
            index=pd.Index([], dtype="object", name="str_idx"),
        )
        result = generate_mock_data(meta, n_rows=3)

        assert result.index.name == "str_idx"
        assert result.index.dtype == np.dtype("object")
        assert all(
            isinstance(idx, str) and idx.startswith("idx_") for idx in result.index
        )

    def test_datetime_index(self):
        """Test datetime index generation."""
        meta = pd.DataFrame(
            {"col": pd.Series([], dtype="int64")},
            index=pd.DatetimeIndex([], name="date_idx"),
        )
        result = generate_mock_data(meta, n_rows=5)

        assert result.index.name == "date_idx"
        assert pd.api.types.is_datetime64_any_dtype(result.index.dtype)
        assert len(result.index) == 5

    def test_multiindex(self):
        """Test MultiIndex generation."""
        index = pd.MultiIndex.from_tuples([], names=["level1", "level2"])
        meta = pd.DataFrame({"col": pd.Series([], dtype="int64")}, index=index)
        result = generate_mock_data(meta, n_rows=3)

        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["level1", "level2"]
        assert len(result.index) == 3


class TestGenerateMockDataEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_rows(self):
        """Test generation with zero rows."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="int64")})
        result = generate_mock_data(meta, n_rows=0)

        assert len(result) == 0
        assert list(result.columns) == ["col"]
        assert result["col"].dtype == np.dtype("int64")

    def test_large_row_count(self):
        """Test generation with large row count."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="int64")})
        result = generate_mock_data(meta, n_rows=10000)

        assert len(result) == 10000
        assert result["col"].dtype == np.dtype("int64")

    def test_unknown_dtype_fallback(self):
        """Test fallback behavior for unknown dtypes."""
        # Create a DataFrame with a complex dtype that might not be handled
        meta = pd.DataFrame({"col": pd.Series([], dtype="complex128")})

        with patch("megaPLuG.utils.data.logger.warning") as mock_warning:
            result = generate_mock_data(meta, n_rows=3)
            mock_warning.assert_called()

        assert len(result) == 3
        # Should have fallback values
        assert all(
            isinstance(val, str) and val.startswith("default_") for val in result["col"]
        )

    def test_exception_handling_fallback(self):
        """Test exception handling with fallback values."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="int64")})

        # Mock numpy.random to raise exception
        with patch("numpy.random.randint", side_effect=Exception("Mock error")):
            with patch("megaPLuG.utils.data.logger.warning") as mock_warning:
                result = generate_mock_data(meta, n_rows=3)
                mock_warning.assert_called()

        assert len(result) == 3
        # Should have fallback values
        assert all(
            isinstance(val, str) and val.startswith("fallback_")
            for val in result["col"]
        )


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestGenerateMockDataDaskIntegration:
    """Test integration with Dask DataFrames."""

    def test_dask_meta_integration(self):
        """Test using actual Dask DataFrame _meta attribute."""
        # Create a pandas DataFrame
        pdf = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
                "bool_col": [True, False, True],
                "str_col": ["a", "b", "c"],
            }
        )

        # Convert to Dask DataFrame
        ddf = dd.from_pandas(pdf, npartitions=2)

        # Generate mock data from meta
        result = generate_mock_data(ddf._meta, n_rows=5)

        assert len(result) == 5
        # Check that dtypes match original
        for col in pdf.columns:
            assert result[col].dtype == ddf._meta[col].dtype

    def test_complex_dask_schema(self):
        """Test with complex Dask DataFrame schema."""
        pdf = pd.DataFrame(
            {
                "int_nullable": pd.Series([1, 2, None], dtype="Int64"),
                "float_nullable": pd.Series([1.0, 2.0, None], dtype="Float64"),
                "bool_nullable": pd.Series([True, None, False], dtype="boolean"),
                "str_nullable": pd.Series(["a", None, "c"], dtype="string"),
                "category": pd.Categorical(["X", "Y", "X"]),
                "datetime": pd.date_range("2024-01-01", periods=3),
                "timedelta": pd.to_timedelta([1, 2, 3], unit="h"),
            }
        )

        ddf = dd.from_pandas(pdf, npartitions=2)
        result = generate_mock_data(ddf._meta, n_rows=10)

        assert len(result) == 10
        # Verify all complex types are handled correctly
        for col in pdf.columns:
            assert result[col].dtype == ddf._meta[col].dtype


class TestGenerateMockDataParametrized:
    """Parameterized tests for comprehensive coverage."""

    @pytest.mark.parametrize("n_rows", [1, 2, 5, 10, 100])
    def test_various_row_counts(self, n_rows):
        """Test function works with various row counts."""
        meta = pd.DataFrame(
            {
                "int_col": pd.Series([], dtype="int64"),
                "float_col": pd.Series([], dtype="float64"),
            }
        )
        result = generate_mock_data(meta, n_rows=n_rows)
        assert len(result) == n_rows

    @pytest.mark.parametrize("seed", [1, 42, 123, 999])
    def test_seed_consistency(self, seed):
        """Test that same seed always produces same results."""
        meta = pd.DataFrame({"col": pd.Series([], dtype="float64")})
        result1 = generate_mock_data(meta, n_rows=5, seed=seed)
        result2 = generate_mock_data(meta, n_rows=5, seed=seed)
        pd.testing.assert_frame_equal(result1, result2)

    @pytest.mark.parametrize(
        "dtype_combo",
        [
            {"int_col": "int64", "str_col": "object"},
            {"float_col": "Float64", "bool_col": "boolean"},
            {"date_col": "datetime64[ns]", "cat_col": pd.CategoricalDtype(["A", "B"])},
            {
                "int_col": "Int32",
                "float_col": "float32",
                "str_col": "string",
                "bool_col": "bool",
            },
        ],
    )
    def test_mixed_dtype_combinations(self, dtype_combo):
        """Test various combinations of data types."""
        columns = {}
        for col_name, dtype in dtype_combo.items():
            columns[col_name] = pd.Series([], dtype=dtype)

        meta = pd.DataFrame(columns)
        result = generate_mock_data(meta, n_rows=10)

        assert len(result) == 10
        for col_name, expected_dtype in dtype_combo.items():
            if isinstance(expected_dtype, pd.CategoricalDtype):
                assert isinstance(result[col_name].dtype, pd.CategoricalDtype)
            else:
                assert (
                    result[col_name].dtype == pd.Series([], dtype=expected_dtype).dtype
                )
