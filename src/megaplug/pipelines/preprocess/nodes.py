"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

import logging
import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import OneHotEncoder

from megaplug.utils.params import build_df_from_dict

logger = logging.getLogger(__name__)


def get_vius_from_url(url: str, params: dict) -> pd.DataFrame:
    """Get a VIUS dataset from a URL."""
    r = requests.get(url)
    txt = re.sub(r"[\[\]]", "", r.text)
    df = pd.read_csv(StringIO(txt))

    df = df.rename(columns={v: k for k, v in params["col_renamer"].items()})
    df = df.loc[:, list(params["col_renamer"].keys())]

    for col, mult in params["multipliers"].items():
        df[col] = df[col] * mult

    df = df.set_index(params["index_col"]).sort_index()
    return df


def clean_vius_by_home_base_state(vius: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Clean the VIUS VMT by home base state table."""
    vius = vius.drop(index=params["drop_idx_values"])
    orig_idx = vius.index.names
    vius = vius.reset_index()
    for old, new in params["replace_values"].items():
        vius = vius.replace(old, new)
    vius = vius.set_index(orig_idx)
    return vius


def clean_vius_by_weight_class(weights: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Clean the VIUS VMT by weight class table."""
    weights = weights.drop(index=params["drop_idx_values"])
    return weights


def build_vius_scaling_totals(vius: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Build a scaling factor dependent on home base state and weight class."""
    corresp_hb = build_df_from_dict(
        d=params["home_base_corresp"]["values"],
        id_cols=params["home_base_corresp"]["id_columns"],
        value_col="home_base_code",
    )
    corresp_cab = build_df_from_dict(
        d=params["cab_type_corresp"]["values"],
        id_cols=params["cab_type_corresp"]["id_columns"],
        value_col="cab_type_code",
    )
    # TODO: Then impute Day Cab and Sleeper Cab for unreported using reported ratio
    scaler = vius.rename(columns={v: k for k, v in params["col_renamer"].items()})
    scaler = scaler.merge(corresp_hb, how="left", on=params["home_source_col"])
    scaler = scaler.merge(corresp_cab, how="left", on=params["cab_source_col"])

    # Set up selection series
    enc = OneHotEncoder(sparse_output=False)
    ohot = enc.fit_transform(scaler.loc[:, ["home_base_code"]])
    ohot = pd.DataFrame(ohot, columns=enc.categories_[0], dtype=bool)
    is_reported = ~ohot["Not Reported"] & ~ohot["Not In Use"]

    # Set up grouping series
    scaler.loc[ohot["Home Base in Register State"], params["id_cols"]["region"]] = (
        scaler.loc[ohot["Home Base in Register State"], "reg_state"]
    )
    scaler.loc[ohot["No Home Base"], params["id_cols"]["region"]] = scaler.loc[
        ohot["No Home Base"], "home_base_code"
    ]

    # Calculate weight adjustments
    weights = scaler[params["totals_col"]]
    p_home_base_known_g_has_home_base = (
        ohot["Home Base in Register State"] * weights
    ).sum() / (
        (ohot["Home Base in Register State"] | ohot["Home Base not in Register State"])
        * weights
    ).sum()
    p_is_reported = (is_reported * weights).sum() / (
        ~ohot["Not In Use"] * weights
    ).sum()

    scaler.loc[is_reported, "reported_mult"] = 1 / p_is_reported
    scaler["reported_mult"] = scaler["reported_mult"].fillna(1.0)
    scaler.loc[ohot["Home Base in Register State"], "specific_mult"] = (
        1 / p_home_base_known_g_has_home_base
    )
    scaler["specific_mult"] = scaler["specific_mult"].fillna(1.0)

    drop_idx = scaler.loc[
        ~is_reported | ohot["Home Base not in Register State"] | ohot["Not In Use"]
    ].index
    reduced = scaler.drop(drop_idx)
    reduced[params["totals_col"]] = (
        reduced[params["totals_col"]]
        * reduced["reported_mult"]
        * reduced["specific_mult"]
    )

    orig_wgt = scaler.loc[~ohot["Not In Use"], params["totals_col"]].sum()
    new_wgt = reduced[params["totals_col"]].sum()

    if not np.isclose(orig_wgt, new_wgt):
        raise RuntimeError(
            "Redistributed total weight does not match original total weight."
        )

    id_cols = list(params["id_cols"].values())
    totals = scaler.groupby(id_cols)[params["totals_col"]].sum()
    totals = totals.reset_index()
    return totals
