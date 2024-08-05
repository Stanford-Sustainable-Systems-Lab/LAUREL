import h3.api.numpy_int as h3
import pandas as pd
from shapely import Polygon


def str_to_h3(s: pd.Series) -> pd.Series:
    return s.transform(h3.str_to_int)


def h3_to_poly(h: int) -> Polygon:
    bnd = h3.cell_to_boundary(h)
    # h3 outputs geometries in lat-lon format, but the convention in WGS84 is lon-lat
    bnd_flip = [(x, y) for y, x in bnd]
    poly = Polygon(bnd_flip)
    return poly
