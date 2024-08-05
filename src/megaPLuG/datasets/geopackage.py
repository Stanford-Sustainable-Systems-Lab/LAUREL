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

    DEFAULT_LAYER_NAME = "default_layer_name"
    GEOPACKAGE_DRIVER_CODE = "GPKG"

    def __init__(self, filepath: str):
        """Creates a new instance of GeoPackageDataset to load / save geographic data
        for given filepath.

        Args:
            filepath: The location of the GeoPackage file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> gpd.GeoDataFrame:
        """Loads data from the GeoPackage file.

        Returns:
            Data from the GeoPackage file as a geopandas.GeoDataFrame
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path, mode="r") as f:
            gdf = gpd.read_file(f)
            return gdf

    def _save(
        self,
        data: gpd.GeoDataFrame,
        index: bool = False,
        layer: str = None,
    ) -> None:
        """Saves geographic data to the specified filepath."""
        save_path = get_filepath_str(self._filepath, self._protocol)
        if layer is None:
            layer = self.DEFAULT_LAYER_NAME
        with self._fs.open(save_path, mode="wb") as f:
            data.to_file(
                f, driver=self.GEOPACKAGE_DRIVER_CODE, index=index, layer=layer
            )

    def _describe(self) -> dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)
