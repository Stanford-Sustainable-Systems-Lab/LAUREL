"""DirectoryPartitionedDataset for handling partitioned datasets where each partition is a directory"""

from __future__ import annotations

from pathlib import PurePath
from typing import Any

from kedro_datasets.partitions import PartitionedDataset


class DirectoryPartitionedDataset(PartitionedDataset):
    """DirectoryPartitionedDataset extends PartitionedDataset to work with directory-based partitions.

    This dataset treats directories (not individual files) as partitions, which is useful
    for datasets like Dask Parquet, GeoPandas Parquet, or any other format that stores
    data as directories containing multiple files.

    Unlike the standard PartitionedDataset which treats each file as a partition, this
    dataset treats each directory as a partition, making it suitable for:
    - Dask Parquet datasets (directories with part.*.parquet files)
    - GeoPandas Parquet datasets
    - Any custom dataset that stores partitioned data as directories

    Example usage in catalog.yml:
    ```yaml
    # With Dask Parquet
    my_dask_partitioned_data:
      type: laurel.datasets.directory_partitioned.DirectoryPartitionedDataset
      path: data/07_model_output/my_dask_data
      dataset:
        type: dask.ParquetDataset
        save_args:
          write_index: False
          engine: pyarrow
        load_args:
          engine: pyarrow

    # With GeoPandas Parquet
    my_geo_partitioned_data:
      type: laurel.datasets.directory_partitioned.DirectoryPartitionedDataset
      path: data/07_model_output/my_geo_data
      dataset:
        type: laurel.datasets.geoparquet.GeoParquetDataset
        save_args:
          write_index: False
        load_args:
          engine: pyarrow
    ```

    The dataset will create/read directory structures like:
    ```
    data/07_model_output/my_data/
    ├── partition1/
    │   ├── part.0.parquet
    │   ├── part.1.parquet
    │   └── _metadata
    └── partition2/
        ├── part.0.parquet
        ├── part.1.parquet
        └── _metadata
    ```

    On load, returns a dict with partition IDs as keys and lazy load functions as values,
    exactly like the standard PartitionedDataset.

    On save, expects a dict with partition IDs as keys and data as values.
    """

    def __init__(
        self,
        *,
        path: str,
        dataset: str | type | dict[str, Any],
        credentials: dict[str, Any] | None = None,
        load_args: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        overwrite: bool = False,
        save_lazily: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Creates a new instance of DirectoryPartitionedDataset.

        Args:
            path: Path to the directory containing partitioned data directories.
            dataset: Underlying dataset definition for each partition directory.
                This can be any dataset type that can handle directory-based data.
                Accepted formats are:
                a) object of a class that inherits from AbstractDataset
                b) a string representing a fully qualified class name to such class
                c) a dictionary with 'type' key pointing to a string from b),
                   other keys are passed to the dataset initializer.
            credentials: Protocol-specific options passed to fsspec.filesystem.
            load_args: Keyword arguments passed to filesystem find() method.
            fs_args: Extra arguments for underlying filesystem class constructor.
            overwrite: If True, existing partitions will be removed before saving.
            save_lazily: Enable/disable lazy saving (default True).
            metadata: Any arbitrary metadata (ignored by Kedro).
        """
        # Initialize parent with filename_suffix="" to avoid file-based filtering
        # The underlying dataset will handle the actual file operations
        super().__init__(
            path=path,
            dataset=dataset,
            filepath_arg="filepath",  # Standard argument name for dataset filepath
            filename_suffix="",  # Don't filter by file extension - we work with directories
            credentials=credentials,
            load_args=load_args,
            fs_args=fs_args,
            overwrite=overwrite,
            save_lazily=save_lazily,
            metadata=metadata,
        )

    def _list_partitions(self) -> list[str]:  # noqa: PLR0912
        """List all leaf directories as partitions (supports hierarchical partitioning).

        Overrides parent method to recursively find bottom-level directories.
        Each leaf directory (directory with files but no subdirectories) is treated
        as a partition, enabling hierarchical partition structures.

        Returns:
            List of paths to leaf partition directories.
        """
        try:
            # Get all file paths recursively
            all_paths = self._filesystem.find(self._normalized_path, **self._load_args)
        except (FileNotFoundError, OSError):
            return []

        # Separate directories and files by checking what exists
        all_directories = set()
        files = set()

        for path in all_paths:
            try:
                # Check if this path is a directory
                if self._filesystem.isdir(path):
                    all_directories.add(path)
                else:
                    # It's a file, add its parent directories to our directory set
                    files.add(path)
                    parent_path = str(PurePath(path).parent)
                    # Add all parent directories up to the base path
                    while parent_path != self._normalized_path and parent_path != str(
                        PurePath(parent_path).parent
                    ):
                        all_directories.add(parent_path)
                        parent_path = str(PurePath(parent_path).parent)
            except (FileNotFoundError, OSError):
                continue

        # For each directory, determine if it's a leaf directory
        leaf_directories = []
        for directory in all_directories:
            try:
                # Get directory contents using fsspec (works with remote filesystems)
                dir_contents = self._filesystem.listdir(directory, detail=False)

                # Use fsspec to check if each item is a file or directory
                has_files = False
                has_subdirs = False

                for item_path in dir_contents:
                    if self._filesystem.isfile(item_path):
                        has_files = True
                    elif self._filesystem.isdir(item_path):
                        has_subdirs = True

                    # Early exit if we found both types
                    if has_files and has_subdirs:
                        break

                # Leaf directory: has files but no subdirectories
                if has_files and not has_subdirs:
                    leaf_directories.append(directory)

            except (FileNotFoundError, OSError, PermissionError):
                continue

        return sorted(leaf_directories)

    def _path_to_partition(self, path: str) -> str:
        """Convert a partition directory path to a partition ID.

        Supports hierarchical partitioning by returning the full relative path
        from the base directory to the partition directory.

        Args:
            path: Full path to the partition directory (can be nested).

        Returns:
            Partition ID (relative path from base directory, e.g., "year=2023/month=01/day=15").
        """
        # Get the normalized base path (use fsspec for protocol stripping)
        base_path_str = self._filesystem._strip_protocol(self._normalized_path)

        # Use pathlib for path operations
        path_obj = PurePath(path)
        base_path_obj = PurePath(base_path_str)

        try:
            # Extract the relative path from base directory
            relative_path = path_obj.relative_to(base_path_obj)
            return str(relative_path)
        except ValueError:
            # Fallback: use the directory name if relative_to fails
            return path_obj.name

    def _partition_to_path(self, partition_id: str) -> str:
        """Convert a partition ID to a partition directory path.

        Supports hierarchical partitioning by handling nested partition IDs.

        Args:
            partition_id: The partition identifier (can be nested path like "year=2023/month=01").

        Returns:
            Full path to the partition directory.
        """
        # Use pathlib for path operations
        base_path = PurePath(self._path)
        partition_path = PurePath(partition_id)

        # Join base path with partition ID to create directory path
        full_path = base_path / partition_path
        return str(full_path)
