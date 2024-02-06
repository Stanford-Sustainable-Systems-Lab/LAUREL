"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""
import dask.dataframe as dd

def preprocess_navistar(navistar, params):
    """Preprocess navistar data as a dask DataFrame object."""
    navistar = navistar.categorize(params["category_columns"])

    for col in params["time_columns"]:
        navistar[col] = dd.to_datetime(navistar[col], utc=True)

    return navistar.compute()