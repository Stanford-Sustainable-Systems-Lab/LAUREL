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
            if isinstance(self._load_args["layer"], list):
                raise NotImplementedError(f"{type(self)} can only load a single layer.")
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self) -> gpd.GeoDataFrame:
        """Loads data from the GeoPackage file.

        Returns:
            Data from the GeoPackage file as a geopandas.GeoDataFrame
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        gdf = gpd.read_file(load_path, **self._load_args)
        return gdf

    def _save(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> None:
        """Saves geographic data to the specified filepath.

        If the "layer" argument is a string, then the GeoDataFrame will be saved to a
        layer with the name given by "layer" using the currently set geometry.

        If the "layer" argument is a dictionary, then the GeoDataFrame will be saved
        to a layer with the name given by the key of layer and the geometry in the
        column of the GeoDataFrame indexed by the value of the dict.
        """
        save_path = get_filepath_str(self._filepath, self._protocol)
        save_args = deepcopy(self._save_args)
        layers = save_args.pop("layer")
        if isinstance(layers, str):  # Save out the GeoDataFrame using layer as the name
            gdf.to_file(save_path, layer=layers, **save_args)
        elif isinstance(layers, dict):  # Save out different layers for each geometry
            geosers = {name: gdf[geom_col] for name, geom_col in layers.items()}
            gdf = gdf.drop(columns=list(layers.values()))
            for name, geom in geosers.items():
                gdf = gdf.set_geometry(geom)
                gdf.to_file(save_path, layer=name, **save_args)
        else:
            raise NotImplementedError()

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath,
            "protocol": self._protocol,
            "load_args": self._load_args,
            "save_args": self._save_args,
        }
