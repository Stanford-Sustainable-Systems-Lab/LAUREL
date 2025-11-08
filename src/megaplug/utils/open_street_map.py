import re
from pathlib import Path

import geopandas as gpd
import osmium

from megaplug.utils.h3 import H3_CRS


class RegexTagFilter:
    def __init__(self, tag: str, pattern: str):
        """A pattern match makes the object be kept."""
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
    """Get OpenStreetMap features whose names match any of the regex patterns."""
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
