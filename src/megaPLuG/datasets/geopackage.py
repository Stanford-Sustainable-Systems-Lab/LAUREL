from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any

import fsspec
import geopandas as gpd
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class GeoPackageDataset(AbstractDataset[gpd.GeoDataFrame, gpd.GeoDataFrame]):
    """``GeoPackageDataset`` loads / saves geographic data from a given filepath as
    `geopandas.GeoDataFrame` using the GeoPackage format.

    Example:
    ::

        >>> GeoPackageDataset(filepath='/geo/file/path.gpkg')
    """

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {
        "index": False,
        "layer": "default_layer",
        "driver": "GPKG",
    }

    def __init__(
        self,
        filepath: str,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
    ):
        """Creates a new instance of ``GeoPackageDataset`` pointing to a concrete GeoPackage file
        on a specific filesystem.

        Args:
            filepath: Filepath in POSIX format to a GeoPackage file prefixed with a protocol like `s3://`.
                If prefix is not provided, `file` protocol (local filesystem) will be used.
                The prefix should be any protocol supported by ``fsspec``.
            load_args: GeoPandas options for loading GeoPackage files.
                Here you can find all available arguments:
                https://geopandas.org/en/stable/docs/reference/api/geopandas.read_file.html
                All defaults are preserved.
            save_args: GeoPandas options for saving GeoPackage files.
                Here you can find all available arguments:
                https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_file.html
                All defaults are preserved, but "index", which is set to False.
            metadata: Any Any arbitrary metadata.
                This is ignored by Kedro, but may be consumed by users or external plugins.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

        # Handle default load and save arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self) -> gpd.GeoDataFrame:
        """Loads data from the GeoPackage file.

        Returns:
            Data from the GeoPackage file as a geopandas.GeoDataFrame
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path, mode="r") as f:
            gdf = gpd.read_file(f, **self._load_args)
            return gdf

    def _save(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> None:
        """Saves geographic data to the specified filepath."""
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, mode="wb") as f:
            gdf.to_file(f, **self._save_args)

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath,
            "protocol": self._protocol,
            "load_args": self._load_args,
            "save_args": self._save_args,
        }
