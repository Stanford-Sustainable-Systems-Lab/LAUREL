"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""
import dask.dataframe as dd

def format_navistar_columns(navistar, params):
    """Preprocess navistar data as a dask DataFrame object."""
    navistar = navistar.categorize(params["category_columns"])

    for col in params["time_columns"]:
        navistar[col] = dd.to_datetime(navistar[col], utc=True)

    return navistar.compute()


def build_h3_polygons(us_outline):
    """Build H3 Scale 8 polygons for full continental U.S."""
    return h3_polygons


def clean_vius(vius):
    """Remove unnecessary VIUS columns and calculate new ones, if necessary."""
    return vius