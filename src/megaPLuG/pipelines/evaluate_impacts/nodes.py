"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""
import dask.dataframe as dd
import geopandas as gpd
import matplotlib
import pandas as pd


def aggregate_regional_loads(
    sessions: dd.DataFrame,
    grid_regions: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Aggregate loads within grid impact regions."""
    raise NotImplementedError()


def plot_peak_load_evolution(
    vehicle_load: pd.DataFrame,
    baseline_load: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Plot baseline loads compared to baseline plus vehicles loads."""
    raise NotImplementedError()


def plot_hourly_load(
    vehicle_load: pd.DataFrame,
    baseline_load: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Plot baseline loads compared to baseline plus vehicles loads."""
    raise NotImplementedError()
