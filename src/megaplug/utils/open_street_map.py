"""OpenStreetMap PBF extraction utilities using the ``osmium`` library.

Provides helpers for filtering OSM PBF files by tag patterns and reading the
matching features into a GeoDataFrame.  Used in :mod:`describe_locations` to
extract truck-stop nodes and warehouse polygons from the continental U.S. OSM
extract.

The extraction follows a two-pass strategy:

1. **Filter pass**: scan the PBF with tag-based filters, collect back-references
   (way nodes, relation members), and write matching objects to a temporary PBF
   via ``osmium.BackReferenceWriter``.
2. **Read pass**: re-scan the temporary PBF with location resolution
   (``with_locations()``) and the ``GeoInterfaceFilter`` to materialise
   geometries, then load into a GeoDataFrame.

This two-pass approach is required because OSM PBFs store node coordinates
separately from way/relation references; the back-reference writer resolves
those links without loading the full planet file into memory.
"""

import re
from pathlib import Path

import geopandas as gpd
import osmium

from megaplug.utils.h3 import H3_CRS


class RegexTagFilter:
    """``osmium`` handler that retains OSM objects whose tag value matches a regex.

    Implements the ``osmium.BaseHandler`` interface (``node``, ``way``,
    ``relation`` methods) so it can be chained via
    ``osmium.FileProcessor.with_filter()``.  An object passes the filter if its
    tag is present **and** the tag value matches the compiled pattern.  Objects
    with the tag absent are excluded.

    Args:
        tag: OSM tag key to inspect (e.g. ``"amenity"``, ``"name"``).
        pattern: Regular expression pattern; matching is case-insensitive.
            A pattern *match* causes the object to be *kept*.
    """

    def __init__(self, tag: str, pattern: str):
        """Compile the regex and store the tag key."""
        self.tag = tag
        self.pattern = re.compile(pattern, re.IGNORECASE)

    def _filter_regex(self, tag_value: str) -> bool:
        """Return True if the object should be filtered out, false if it should be kept."""
        cleaned_tag_value = tag_value.strip()
        return not self.pattern.search(cleaned_tag_value)

    def _process_obj(self, n) -> bool:
        tag = n.tags.get(self.tag, None)
        if tag:
            return self._filter_regex(tag)
        else:
            return True

    def node(self, n):
        return self._process_obj(n)

    def way(self, n):
        return self._process_obj(n)

    def relation(self, n):
        return self._process_obj(n)


def processor_factory(
    pbf_path: Path,
    filters: list[osmium.BaseHandler],
    with_locations=False,
    **geo_int_kwargs,
) -> osmium.FileProcessor:
    """Build a configured ``osmium.FileProcessor`` with filters and optional geometry.

    Args:
        pbf_path: Path to the input OSM PBF file.
        filters: List of ``osmium.BaseHandler``-compatible filter objects applied
            in order via ``with_filter()``.
        with_locations: If ``True``, attaches a location index (needed to resolve
            way/relation node coordinates) and appends a
            ``GeoInterfaceFilter``.
        **geo_int_kwargs: Additional keyword arguments forwarded to
            ``osmium.filter.GeoInterfaceFilter`` (e.g. ``tags``).

    Returns:
        Configured ``osmium.FileProcessor`` ready for iteration.
    """
    fp = osmium.FileProcessor(pbf_path)
    for filt in filters:
        fp = fp.with_filter(filt)
    if with_locations:
        fp = fp.with_locations()
        fp = fp.with_filter(osmium.filter.GeoInterfaceFilter(**geo_int_kwargs))
    return fp


def get_gdf_from_filtered_osm(
    osm_path: Path, filters: list[osmium.BaseHandler], tags: list[str], temp_path: Path
) -> gpd.GeoDataFrame:
    """Extract tag-filtered OSM features from a PBF file into a GeoDataFrame.

    Uses the two-pass strategy described in the module docstring: first writes
    matching objects (with back-references resolved) to ``temp_path``, then
    re-reads with location resolution to materialise geometries.

    Args:
        osm_path: Path to the source OSM PBF file (e.g. the continental U.S.
            extract).
        filters: List of filter handlers (e.g. :class:`RegexTagFilter` instances)
            applied to both passes.
        tags: List of OSM tag keys to include as columns in the output
            GeoDataFrame (forwarded to ``GeoInterfaceFilter``).
        temp_path: Writable path for the intermediate filtered PBF.  Overwritten
            if it already exists.

    Returns:
        GeoDataFrame of matching features in ``EPSG:4326``.
    """
    # Get all objects of interest based on tags, and write them and their back-references to disk
    with osmium.BackReferenceWriter(
        temp_path, ref_src=osm_path, overwrite=True
    ) as writer:
        for o in processor_factory(pbf_path=osm_path, filters=filters):
            writer.add(o)

    # Reading selected objects into a GeoDataFrame
    feats = processor_factory(
        pbf_path=temp_path, filters=filters, with_locations=True, tags=tags
    )
    gdf = gpd.GeoDataFrame.from_features(feats)
    gdf = gdf.set_crs(H3_CRS)
    return gdf
