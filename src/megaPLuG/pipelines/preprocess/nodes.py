"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

import logging

import dask.dataframe as dd

logger = logging.getLogger(__name__)


def format_trips_columns(trips, params):
    """Preprocess trips data columns."""
    trips = trips.categorize(params["category_columns"])

    for col in params["time_columns"]:
        trips[col] = dd.to_datetime(trips[col], utc=True)

    return trips


def set_trips_index(trips, params):
    """Set index of trips data."""
    if params["debug_subsample"]["active"]:
        trips = trips.loc[0 : params["debug_subsample"]["n"]].compute()
        trips = dd.from_pandas(trips, npartitions=2)

    trips = trips.sort_values(by=params["sort_column_order"])

    trips["temp"] = 1
    trips[params["index_column"]] = trips["temp"].cumsum()
    trips = trips.drop(columns=["temp"])

    logger.info("Starting indexing compute")
    trips = trips.set_index(params["index_column"], sorted=True)
    return trips


def build_h3_polygons(us_outline):
    """Build H3 Scale 8 polygons for full continental U.S.

    Might be unnecessary if we can build polygons quickly on the fly, but this
    depends on H3.
    """
    raise NotImplementedError()


def clean_vius(vius):
    """Remove unnecessary VIUS columns and calculate new ones, if necessary."""
    raise NotImplementedError()
