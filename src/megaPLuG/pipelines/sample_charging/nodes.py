"""
This is a boilerplate pipeline 'sample_charging'
generated using Kedro 0.19.1
"""

import dask.dataframe as dd
import pandas as pd

from megaPLuG.models.charging_pgm import DistanceTimePGM


def calc_model_conditioning(
    adopt: pd.DataFrame,
    vius: pd.DataFrame,
    params: dict,
) -> dict:
    """Calculate PGM conditioning dictionary from adoption scenario and observed
    patterns as of 2022."""
    raise NotImplementedError()


def sample_charging_sessions(
    model: DistanceTimePGM,
    conds: dict,
    params: dict,
) -> dd.DataFrame:
    """Sample the model to get charging sessions out."""
    raise NotImplementedError()
