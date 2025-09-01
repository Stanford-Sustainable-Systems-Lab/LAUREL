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
      type: megaPLuG.datasets.directory_partitioned.DirectoryPartitionedDataset
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
      type: megaPLuG.datasets.directory_partitioned.DirectoryPartitionedDataset
      path: data/07_model_output/my_geo_data
      dataset:
        type: megaPLuG.datasets.geoparquet.GeoParquetDataset
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

    def _list_partitions(self) -> list[str]:
        """List all partition directories.

        Overrides parent method to return directories instead of files.

        Returns:
            List of paths to partition directories.
        """
        # Use listdir to get immediate children, then filter for directories
        try:
            items = self._filesystem.listdir(self._normalized_path, detail=True)
        except (FileNotFoundError, OSError):
            return []

        # Filter to only include directories that contain files
        partitions = []
        for item in items:
            if item["type"] == "directory":
                dir_path = item["name"]
                try:
                    # Check if directory has content (not empty)
                    contents = self._filesystem.listdir(dir_path, detail=False)
                    if contents:  # Directory exists and has content
                        partitions.append(dir_path)
                except (FileNotFoundError, OSError):
                    # Can't access directory, skip
                    continue

        return sorted(partitions)

    def _path_to_partition(self, path: str) -> str:
        """Convert a partition directory path to a partition ID.

        Args:
            path: Full path to the partition directory.

        Returns:
            Partition ID (directory name relative to base path).
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

        Args:
            partition_id: The partition identifier.

        Returns:
            Full path to the partition directory.
        """
        # Use pathlib for path operations
        base_path = PurePath(self._path)
        partition_path = PurePath(partition_id)

        # Join base path with partition ID to create directory path
        full_path = base_path / partition_path
        return str(full_path)
