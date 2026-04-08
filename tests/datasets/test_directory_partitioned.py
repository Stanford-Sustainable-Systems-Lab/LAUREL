"""Comprehensive test suite for DirectoryPartitionedDataset class."""

import shutil
import tempfile
from pathlib import Path, PurePath
from unittest.mock import patch

import pandas as pd
import pytest
from kedro.io.core import DatasetError
from kedro_datasets.partitions import PartitionedDataset

# Import the module under test
from laurel.datasets.directory_partitioned import DirectoryPartitionedDataset

try:
    import dask.dataframe as dd

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


class TestDirectoryPartitionedDatasetConstructor:
    """Test DirectoryPartitionedDataset constructor and initialization."""

    def test_init_with_dict_dataset(self, tmp_path):
        """Test initialization with dataset as dictionary."""
        dataset_config = {
            "type": "pandas.ParquetDataset",
            "save_args": {"engine": "pyarrow"},
            "load_args": {"engine": "pyarrow"},
        }

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset=dataset_config
        )

        assert dataset._path == str(tmp_path)
        # Access the dataset type and config from parent class
        assert dataset._dataset_type.__name__ == "ParquetDataset"
        assert dataset._dataset_config["save_args"]["engine"] == "pyarrow"

    def test_init_with_string_dataset(self, tmp_path):
        """Test initialization with dataset as string."""
        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="pandas.ParquetDataset"
        )

        assert dataset._path == str(tmp_path)
        assert dataset._dataset_type.__name__ == "ParquetDataset"

    def test_init_with_credentials(self, tmp_path):
        """Test initialization with credentials."""
        credentials = {
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
        }

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="dask.ParquetDataset", credentials=credentials
        )

        assert dataset._credentials == credentials

    def test_init_with_all_parameters(self, tmp_path):
        """Test initialization with all parameters."""
        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path),
            dataset={"type": "dask.ParquetDataset"},
            credentials={"key": "value"},
            load_args={"param1": "value1"},
            fs_args={"param2": "value2"},
            overwrite=True,
            save_lazily=False,
            metadata={"description": "test dataset"},
        )

        assert dataset._path == str(tmp_path)
        assert dataset._overwrite is True
        assert dataset._save_lazily is False
        assert dataset.metadata["description"] == "test dataset"

    def test_inheritance(self, tmp_path):
        """Test that DirectoryPartitionedDataset inherits from PartitionedDataset."""
        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="pandas.ParquetDataset"
        )

        assert isinstance(dataset, PartitionedDataset)
        assert hasattr(dataset, "load")
        assert hasattr(dataset, "save")
        assert hasattr(dataset, "exists")


class TestDirectoryPartitionedDatasetPartitionDiscovery:
    """Test partition discovery logic (_list_partitions method)."""

    def test_list_partitions_empty_directory(self, tmp_path):
        """Test partition listing with empty directory."""
        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="pandas.ParquetDataset"
        )

        partitions = dataset._list_partitions()
        assert partitions == []

    def test_list_partitions_with_directories(self, tmp_path):
        """Test partition listing with valid directories."""
        # Create directories with content
        (tmp_path / "partition1").mkdir()
        (tmp_path / "partition2").mkdir()
        (tmp_path / "partition1" / "data.parquet").write_text("test")
        (tmp_path / "partition2" / "data.parquet").write_text("test")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="pandas.ParquetDataset"
        )

        partitions = dataset._list_partitions()
        assert len(partitions) == 2

        # Extract directory names from full paths
        partition_names = {PurePath(p).name for p in partitions}
        assert partition_names == {"partition1", "partition2"}

    def test_list_partitions_ignores_empty_directories(self, tmp_path):
        """Test that empty directories are ignored."""
        # Create directories - one with content, one empty
        (tmp_path / "partition1").mkdir()
        (tmp_path / "empty_partition").mkdir()
        (tmp_path / "partition1" / "data.parquet").write_text("test")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="pandas.ParquetDataset"
        )

        partitions = dataset._list_partitions()
        assert len(partitions) == 1
        assert PurePath(partitions[0]).name == "partition1"

    def test_list_partitions_ignores_files(self, tmp_path):
        """Test that files in base directory are ignored."""
        # Create both directories and files
        (tmp_path / "partition1").mkdir()
        (tmp_path / "some_file.txt").write_text("test")
        (tmp_path / "partition1" / "data.parquet").write_text("test")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="pandas.ParquetDataset"
        )

        partitions = dataset._list_partitions()
        assert len(partitions) == 1
        assert PurePath(partitions[0]).name == "partition1"

    def test_list_partitions_nonexistent_path(self, tmp_path):
        """Test partition listing with non-existent path."""
        nonexistent_path = tmp_path / "nonexistent"

        dataset = DirectoryPartitionedDataset(
            path=str(nonexistent_path), dataset="pandas.ParquetDataset"
        )

        partitions = dataset._list_partitions()
        assert partitions == []

    def test_list_partitions_sorted(self, tmp_path):
        """Test that partitions are returned in sorted order."""
        # Create directories in non-alphabetical order
        for name in ["partition_c", "partition_a", "partition_b"]:
            (tmp_path / name).mkdir()
            (tmp_path / name / "data.parquet").write_text("test")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="pandas.ParquetDataset"
        )

        partitions = dataset._list_partitions()
        partition_names = [PurePath(p).name for p in partitions]
        assert partition_names == ["partition_a", "partition_b", "partition_c"]


class TestDirectoryPartitionedDatasetPathConversion:
    """Test path conversion methods (_path_to_partition and _partition_to_path)."""

    @pytest.fixture
    def dataset(self, tmp_path):
        """Fixture for DirectoryPartitionedDataset instance."""
        return DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="pandas.ParquetDataset"
        )

    def test_path_to_partition_simple(self, dataset, tmp_path):
        """Test conversion from path to partition ID."""
        partition_path = str(tmp_path / "partition1")
        partition_id = dataset._path_to_partition(partition_path)
        assert partition_id == "partition1"

    def test_path_to_partition_nested(self, dataset, tmp_path):
        """Test conversion with nested partition structure."""
        partition_path = str(tmp_path / "group1" / "partition1")
        partition_id = dataset._path_to_partition(partition_path)
        assert partition_id == str(PurePath("group1") / "partition1")

    def test_path_to_partition_fallback(self, dataset):
        """Test fallback behavior when relative_to fails."""
        # Use a path that's not relative to the base path
        unrelated_path = "/completely/different/path"
        partition_id = dataset._path_to_partition(unrelated_path)
        assert partition_id == "path"  # Should return the last component

    def test_partition_to_path_simple(self, dataset, tmp_path):
        """Test conversion from partition ID to path."""
        partition_id = "partition1"
        expected_path = str(tmp_path / "partition1")
        actual_path = dataset._partition_to_path(partition_id)
        assert actual_path == expected_path

    def test_partition_to_path_nested(self, dataset, tmp_path):
        """Test conversion with nested partition ID."""
        partition_id = str(PurePath("group1") / "partition1")
        expected_path = str(tmp_path / "group1" / "partition1")
        actual_path = dataset._partition_to_path(partition_id)
        assert actual_path == expected_path

    def test_path_conversion_roundtrip(self, dataset, tmp_path):
        """Test that path conversion is symmetric."""
        original_partition = "test_partition"

        # Convert partition to path and back
        path = dataset._partition_to_path(original_partition)
        recovered_partition = dataset._path_to_partition(path)

        assert recovered_partition == original_partition

    def test_path_conversion_with_special_characters(self, dataset, tmp_path):
        """Test path conversion with special characters in partition names."""
        # Test with various special characters that are valid in filenames
        special_partitions = ["partition-1", "partition_2", "partition.3"]

        for partition_id in special_partitions:
            path = dataset._partition_to_path(partition_id)
            recovered_partition = dataset._path_to_partition(path)
            assert recovered_partition == partition_id


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestDirectoryPartitionedDatasetDaskIntegration:
    """Test integration with Dask ParquetDataset."""

    @pytest.fixture
    def sample_dask_data(self):
        """Fixture for sample Dask DataFrames."""
        df1 = pd.DataFrame(
            {
                "id": range(100),
                "value": range(100, 200),
                "category": ["A"] * 50 + ["B"] * 50,
            }
        )

        df2 = pd.DataFrame(
            {
                "id": range(100, 200),
                "value": range(200, 300),
                "category": ["C"] * 50 + ["D"] * 50,
            }
        )

        return {
            "partition1": dd.from_pandas(df1, npartitions=2),
            "partition2": dd.from_pandas(df2, npartitions=2),
        }

    def test_save_and_load_dask_data(self, tmp_path, sample_dask_data):
        """Test saving and loading Dask data."""
        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path),
            dataset={
                "type": "dask.ParquetDataset",
                "save_args": {"write_index": False, "engine": "pyarrow"},
                "load_args": {"engine": "pyarrow"},
            },
        )

        # Save data
        dataset.save(sample_dask_data)

        # Verify directory structure
        assert (tmp_path / "partition1").exists()
        assert (tmp_path / "partition2").exists()
        assert len(list((tmp_path / "partition1").glob("*.parquet"))) > 0
        assert len(list((tmp_path / "partition2").glob("*.parquet"))) > 0

        # Load data
        loaded_data = dataset.load()
        assert "partition1" in loaded_data
        assert "partition2" in loaded_data

        # Test lazy loading
        loaded_df1 = loaded_data["partition1"]()
        loaded_df2 = loaded_data["partition2"]()

        assert isinstance(loaded_df1, dd.DataFrame)
        assert isinstance(loaded_df2, dd.DataFrame)
        assert len(loaded_df1.compute()) == 100
        assert len(loaded_df2.compute()) == 100

    def test_exists_method(self, tmp_path, sample_dask_data):
        """Test exists method."""
        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="dask.ParquetDataset"
        )

        # Should not exist initially
        assert not dataset.exists()

        # Should exist after saving data
        dataset.save(sample_dask_data)
        assert dataset.exists()


class TestDirectoryPartitionedDatasetPandasIntegration:
    """Test integration with pandas datasets."""

    @pytest.fixture
    def sample_pandas_data(self):
        """Fixture for sample pandas DataFrames."""
        df1 = pd.DataFrame(
            {
                "id": range(50),
                "value": range(50, 100),
                "name": [f"item_{i}" for i in range(50)],
            }
        )

        df2 = pd.DataFrame(
            {
                "id": range(50, 100),
                "value": range(100, 150),
                "name": [f"item_{i}" for i in range(50, 100)],
            }
        )

        return {"region1": df1, "region2": df2}

    def test_simulated_directory_structure(self, tmp_path, sample_pandas_data):
        """Test loading from pre-created directory structure."""
        # Manually create directory structure
        for partition_id, df in sample_pandas_data.items():
            partition_dir = tmp_path / partition_id
            partition_dir.mkdir()
            df.to_parquet(partition_dir / "data.parquet", engine="pyarrow")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path),
            dataset={
                "type": "pandas.ParquetDataset",
                "save_args": {"engine": "pyarrow"},
                "load_args": {"engine": "pyarrow"},
            },
        )

        # Test load
        loaded_data = dataset.load()
        assert "region1" in loaded_data
        assert "region2" in loaded_data

        # Test lazy loading
        loaded_df1 = loaded_data["region1"]()
        assert isinstance(loaded_df1, pd.DataFrame)
        assert len(loaded_df1) == 50


class TestDirectoryPartitionedDatasetErrorHandling:
    """Test error handling and edge cases."""

    def test_load_empty_dataset_raises_error(self, tmp_path):
        """Test that loading empty dataset raises DatasetError."""
        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="pandas.ParquetDataset"
        )

        with pytest.raises(DatasetError, match="No partitions found"):
            dataset.load()

    def test_invalid_dataset_config(self, tmp_path):
        """Test initialization with invalid dataset configuration."""
        # Invalid dataset type should fail during initialization
        with pytest.raises(DatasetError, match="No module named"):
            DirectoryPartitionedDataset(
                path=str(tmp_path), dataset={"type": "nonexistent.Dataset"}
            )

    def test_permission_error_handling(self, tmp_path):
        """Test handling of permission errors."""
        # Create a directory and make it inaccessible
        restricted_dir = tmp_path / "restricted"
        restricted_dir.mkdir()

        # Create the dataset pointed at the parent directory
        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="pandas.ParquetDataset"
        )

        # Mock filesystem to simulate permission error
        with patch.object(dataset._filesystem, "listdir") as mock_listdir:
            mock_listdir.side_effect = OSError("Permission denied")

            partitions = dataset._list_partitions()
            assert partitions == []

    def test_corrupted_directory_structure(self, tmp_path):
        """Test handling of corrupted directory structure."""
        # Create a directory that looks like a partition but has issues
        partition_dir = tmp_path / "corrupted_partition"
        partition_dir.mkdir()

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="pandas.ParquetDataset"
        )

        # Mock to simulate listdir failing on the partition directory
        original_listdir = dataset._filesystem.listdir

        def mock_listdir(path, detail=False):
            if path.endswith("corrupted_partition"):
                raise OSError("Directory corrupted")
            return original_listdir(path, detail)

        with patch.object(dataset._filesystem, "listdir", side_effect=mock_listdir):
            partitions = dataset._list_partitions()
            # Should skip the corrupted partition
            assert partitions == []


class TestDirectoryPartitionedDatasetEdgeCases:
    """Test edge cases and performance scenarios."""

    def test_large_number_of_partitions(self, tmp_path):
        """Test with large number of partitions."""
        # Create many partitions
        num_partitions = 100
        for i in range(num_partitions):
            partition_dir = tmp_path / f"partition_{i:03d}"
            partition_dir.mkdir()
            (partition_dir / "data.txt").write_text("test")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset"
        )

        partitions = dataset._list_partitions()
        assert len(partitions) == num_partitions

        # Verify they're sorted
        partition_names = [PurePath(p).name for p in partitions]
        assert partition_names == sorted(partition_names)

    def test_unicode_partition_names(self, tmp_path):
        """Test with unicode characters in partition names."""
        unicode_names = ["測試", "тест", "αβγ", "🚀📊"]

        for name in unicode_names:
            try:
                partition_dir = tmp_path / name
                partition_dir.mkdir()
                (partition_dir / "data.txt").write_text("test")
            except (OSError, UnicodeError):
                # Skip if filesystem doesn't support unicode names
                pytest.skip(f"Filesystem doesn't support unicode name: {name}")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset"
        )

        partitions = dataset._list_partitions()
        partition_names = {PurePath(p).name for p in partitions}

        # Check that all created partitions are found
        for name in unicode_names:
            if (tmp_path / name).exists():
                assert name in partition_names

    def test_deeply_nested_partitions(self, tmp_path):
        """Test with deeply nested partition structure."""
        nested_partition = "level1/level2/level3/partition"
        partition_path = tmp_path / nested_partition
        partition_path.mkdir(parents=True)
        (partition_path / "data.txt").write_text("test")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset"
        )

        # This should find the nested directory
        partitions = dataset._list_partitions()
        assert len(partitions) == 1

        # Test path conversion
        partition_id = dataset._path_to_partition(partitions[0])
        recovered_path = dataset._partition_to_path(partition_id)

        # The paths should be equivalent
        assert PurePath(partitions[0]) == PurePath(recovered_path)


class TestDirectoryPartitionedDatasetHierarchicalPartitioning:
    """Test hierarchical partitioning functionality."""

    def test_hierarchical_date_partitioning(self, tmp_path):
        """Test hierarchical date-based partitioning."""
        # Create hierarchical date structure
        partitions = [
            "year=2023/month=01/day=01/data",
            "year=2023/month=01/day=02/data",
            "year=2023/month=02/day=01/data",
            "year=2024/month=01/day=01/data",
        ]

        for partition in partitions:
            partition_path = tmp_path / partition
            partition_path.mkdir(parents=True)
            (partition_path / "file.txt").write_text("test data")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset"
        )

        # Should find all leaf directories
        found_partitions = dataset._list_partitions()
        assert len(found_partitions) == 4

        # Extract partition IDs
        partition_ids = [dataset._path_to_partition(p) for p in found_partitions]
        expected_ids = [
            "year=2023/month=01/day=01/data",
            "year=2023/month=01/day=02/data",
            "year=2023/month=02/day=01/data",
            "year=2024/month=01/day=01/data",
        ]

        assert set(partition_ids) == set(expected_ids)

    def test_mixed_depth_partitions(self, tmp_path):
        """Test partitions at different depths."""
        # Create partitions at various depths
        shallow_partition = tmp_path / "shallow_data"
        deep_partition = tmp_path / "region=US/state=CA/city=SF/data"
        medium_partition = tmp_path / "category=A/subcategory=1/files"

        shallow_partition.mkdir(parents=True)
        deep_partition.mkdir(parents=True)
        medium_partition.mkdir(parents=True)

        # Add files to make them leaf directories
        (shallow_partition / "file1.txt").write_text("data1")
        (deep_partition / "file2.txt").write_text("data2")
        (medium_partition / "file3.txt").write_text("data3")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset"
        )

        partitions = dataset._list_partitions()
        assert len(partitions) == 3

        # Check partition IDs
        partition_ids = set(dataset._path_to_partition(p) for p in partitions)
        expected_ids = {
            "shallow_data",
            "region=US/state=CA/city=SF/data",
            "category=A/subcategory=1/files",
        }

        assert partition_ids == expected_ids

    def test_ignores_intermediate_directories(self, tmp_path):
        """Test that intermediate directories (non-leaf) are ignored."""
        # Create structure where only leaf directories should be partitions
        structure = {
            "country=US/state=CA": False,  # Intermediate directory
            "country=US/state=CA/city=SF/data": True,  # Leaf partition
            "country=US/state=NY": False,  # Intermediate directory
            "country=US/state=NY/city=NYC/data": True,  # Leaf partition
            "country=UK": False,  # Intermediate directory
            "country=UK/region=London/data": True,  # Leaf partition
        }

        for path, is_leaf in structure.items():
            dir_path = tmp_path / path
            dir_path.mkdir(parents=True)
            if is_leaf:
                (dir_path / "file.txt").write_text("data")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset"
        )

        partitions = dataset._list_partitions()

        # Should only find the 3 leaf directories
        assert len(partitions) == 3

        partition_ids = set(dataset._path_to_partition(p) for p in partitions)
        expected_leaf_ids = {
            "country=US/state=CA/city=SF/data",
            "country=US/state=NY/city=NYC/data",
            "country=UK/region=London/data",
        }

        assert partition_ids == expected_leaf_ids

    def test_hierarchical_path_conversion_roundtrip(self, tmp_path):
        """Test that hierarchical path conversion is symmetric."""
        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset"
        )

        # Test various hierarchical partition IDs
        test_partitions = [
            "simple",
            "level1/level2",
            "year=2023/month=01/day=15/hour=12/data",
            "country=US/state=CA/city=SF/district=SOMA/block=1",
        ]

        for partition_id in test_partitions:
            # Convert partition ID to path and back
            path = dataset._partition_to_path(partition_id)
            recovered_partition = dataset._path_to_partition(path)

            assert recovered_partition == partition_id

    def test_hierarchical_with_special_characters(self, tmp_path):
        """Test hierarchical partitions with special characters."""
        # Create partition with special characters in hierarchy
        partition_structure = (
            "region=North-America/country=US/state=New_York/city=New.York/data"
        )
        partition_path = tmp_path / partition_structure
        partition_path.mkdir(parents=True)
        (partition_path / "file.txt").write_text("test")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset"
        )

        partitions = dataset._list_partitions()
        assert len(partitions) == 1

        partition_id = dataset._path_to_partition(partitions[0])
        assert partition_id == partition_structure

    def test_empty_intermediate_directories_ignored(self, tmp_path):
        """Test that empty intermediate directories don't affect partition discovery."""
        # Create structure with some empty directories
        (tmp_path / "year=2023").mkdir()
        (tmp_path / "year=2023/month=01").mkdir()
        (tmp_path / "year=2023/month=01/day=01").mkdir()  # Empty intermediate

        # Create actual partition with data
        partition_dir = tmp_path / "year=2023/month=01/day=02/data"
        partition_dir.mkdir(parents=True)
        (partition_dir / "file.txt").write_text("data")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset"
        )

        partitions = dataset._list_partitions()

        # Should only find the one partition with actual data
        assert len(partitions) == 1
        partition_id = dataset._path_to_partition(partitions[0])
        assert partition_id == "year=2023/month=01/day=02/data"

    def test_multiple_files_in_leaf_directory(self, tmp_path):
        """Test leaf directories with multiple files."""
        partition_dir = tmp_path / "year=2023/month=01/data"
        partition_dir.mkdir(parents=True)

        # Add multiple files to the leaf directory
        (partition_dir / "file1.txt").write_text("data1")
        (partition_dir / "file2.txt").write_text("data2")
        (partition_dir / "file3.parquet").write_bytes(b"parquet_data")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset"
        )

        partitions = dataset._list_partitions()
        assert len(partitions) == 1

        partition_id = dataset._path_to_partition(partitions[0])
        assert partition_id == "year=2023/month=01/data"

    def test_performance_with_deep_hierarchy(self, tmp_path):
        """Test performance with deeply nested hierarchical structure."""
        # Create a moderately deep structure (5 levels)
        base_structure = "level1/level2/level3/level4/level5"
        num_partitions = 20

        for i in range(num_partitions):
            partition_dir = tmp_path / base_structure / f"partition_{i}"
            partition_dir.mkdir(parents=True)
            (partition_dir / "data.txt").write_text(f"data_{i}")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset"
        )

        partitions = dataset._list_partitions()
        assert len(partitions) == num_partitions

        # Verify all partitions are at the expected depth
        for partition in partitions:
            partition_id = dataset._path_to_partition(partition)
            assert partition_id.startswith(f"{base_structure}/partition_")
            assert partition_id.count("/") == 5  # 5 levels deep

    def test_overwrite_behavior(self, tmp_path):
        """Test overwrite behavior."""
        # Create initial data
        (tmp_path / "partition1").mkdir()
        (tmp_path / "partition1" / "old_data.txt").write_text("old")

        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset", overwrite=True
        )

        # Save new data (would normally overwrite)
        # Note: Actual overwrite behavior is handled by parent class
        # This test just ensures the parameter is passed correctly
        assert dataset._overwrite is True

    def test_save_lazily_parameter(self, tmp_path):
        """Test save_lazily parameter."""
        dataset = DirectoryPartitionedDataset(
            path=str(tmp_path), dataset="text.TextDataset", save_lazily=False
        )

        assert dataset._save_lazily is False


# Test fixtures
@pytest.fixture
def tmp_path():
    """Provide a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)
