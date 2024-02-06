"""
This is a boilerplate pipeline 'electrify_trips'
generated using Kedro 0.19.1
"""
import dask.dataframe as dd
import pandas as pd


def simulate_electrified_trips(
    trips: dd.DataFrame, energy_consump: pd.DataFrame, params: dict
) -> dd.DataFrame:
    """Simulate what would happen if each trip were electrified.

    At first, this will follow the logic of Borlaug et al. Later, I expect it to
    develop into an optimization-based system.
    """
    raise NotImplementedError()
