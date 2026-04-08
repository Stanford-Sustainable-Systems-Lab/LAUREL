"""Kedro pipeline nodes for the ``preprocess`` pipeline (VIUS data preparation).

Prepares the Vehicle Inventory and Use Survey (VIUS) microdata for use as a
fleet-size scaling basis in the ``prepare_totals`` and ``evaluate_impacts``
pipelines.  The VIUS provides nationally representative estimates of the HDT
fleet composition (weight class, home-base state, cab type, operating distance)
that are used to scale telematics observations—which over-represent long-haul
vehicles—to realistic state-level fleet totals.

Pipeline overview
-----------------
1. **get_vius_from_url** — Downloads VIUS microdata from a remote URL, strips
   bracket characters introduced by the Census Bureau's CSV encoding, renames
   columns, and applies unit multipliers.
2. **clean_vius_by_home_base_state** — Drops invalid index rows and normalises
   state-name strings for the VMT-by-home-base-state VIUS table.
3. **clean_vius_by_weight_class** — Drops invalid index rows from the
   VMT-by-weight-class VIUS table.
4. **build_vius_scaling_totals** — Constructs survey-weight-adjusted fleet
   totals stratified by home-base state, weight class, and cab type, spreading
   weight from vehicles with out-of-state or unknown home bases proportionally
   across known-home-base vehicles.

Key design decisions
--------------------
- **Weight redistribution**: VIUS respondents in the "Home Base not in Register
  State" category cannot be assigned to a specific state.  Their survey weight
  is redistributed to "Home Base in Register State" respondents in the same
  operating-distance/weight-class strata proportionally, preserving the total
  weighted fleet count.
- **Reporting-rate correction**: Vehicles that did not report their home base
  (``Not Reported``) or are no longer in use (``Not In Use``) are excluded.
  The remaining vehicles' weights are scaled up by ``1 / p_reported`` so that
  the adjusted total matches the full in-use fleet.
- **Integrity check**: After redistribution the sum of adjusted weights must
  equal the sum of original in-use weights; a ``RuntimeError`` is raised
  otherwise to catch parameter mistakes early.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.

U.S. Census Bureau. Vehicle Inventory and Use Survey (VIUS) 2021.
https://www.census.gov/programs-surveys/vius.html
"""

import logging
import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import OneHotEncoder

from laurel.utils.params import build_df_from_dict

logger = logging.getLogger(__name__)


def get_vius_from_url(url: str, params: dict) -> pd.DataFrame:
    """Download and parse a VIUS microdata table from a remote URL.

    The Census Bureau encodes certain VIUS CSV files with square-bracket
    characters around numeric values; this function strips those characters
    before passing the text to ``pd.read_csv``.  After loading, columns are
    renamed according to the ``col_renamer`` mapping, a subset of columns is
    retained, numeric columns are scaled by optional multipliers, and the
    result is sorted by the specified index column.

    Args:
        url: The full URL to the VIUS CSV file.
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw column names
              to internal names; only keys that appear in the raw data are used.
            - ``multipliers`` (dict[str, float]): mapping from (renamed) column
              name to a scalar multiplier applied element-wise.
            - ``index_col`` (str): column to use as the DataFrame index after
              renaming.

    Returns:
        A ``pd.DataFrame`` indexed by ``params["index_col"]``, sorted
        ascending, with columns limited to those in ``col_renamer`` and
        numeric values scaled by ``multipliers``.
    """
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
    """Remove invalid rows and normalise string values in the VMT-by-home-base-state table.

    Args:
        vius: Raw VIUS VMT-by-home-base-state DataFrame as loaded by
            ``get_vius_from_url``.
        params: Pipeline parameters dict with keys:

            - ``drop_idx_values`` (list): index values to drop (e.g., total
              rows or footnote rows added by the Census Bureau).
            - ``replace_values`` (dict[str, str]): string replacements applied
              DataFrame-wide (e.g., normalising state-name abbreviations).

    Returns:
        A cleaned ``pd.DataFrame`` with invalid rows removed and string values
        normalised.
    """
    vius = vius.drop(index=params["drop_idx_values"])
    orig_idx = vius.index.names
    vius = vius.reset_index()
    for old, new in params["replace_values"].items():
        vius = vius.replace(old, new)
    vius = vius.set_index(orig_idx)
    return vius


def clean_vius_by_weight_class(weights: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Remove invalid rows from the VMT-by-weight-class table.

    Args:
        weights: Raw VIUS VMT-by-weight-class DataFrame.
        params: Pipeline parameters dict with keys:

            - ``drop_idx_values`` (list): index values to drop (e.g., total
              or footnote rows).

    Returns:
        A cleaned ``pd.DataFrame`` with invalid rows removed.
    """
    weights = weights.drop(index=params["drop_idx_values"])
    return weights


def build_vius_scaling_totals(vius: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Construct survey-weight-adjusted HDT fleet totals by state, weight class, and cab type.

    The VIUS microdata contains three home-base categories that require special
    handling before the fleet totals can be used as scaling denominators:

    1. **"Home Base in Register State"**: The vehicle's home base is the same
       state where it is registered.  These records are retained as-is but
       scaled up by ``1 / p_home_base_known`` to account for the vehicles in
       the "Home Base not in Register State" stratum whose home state is
       unknown.
    2. **"Home Base not in Register State"**: The vehicle has a home base in a
       different state.  These records are dropped after their weight has been
       redistributed to stratum-matched "Home Base in Register State" records.
    3. **"No Home Base"**, **"Not Reported"**, **"Not In Use"**: "No Home Base"
       records are retained but not scaled (their home base is treated as
       unknown).  "Not Reported" and "Not In Use" records are excluded
       entirely; their combined weight is recovered via a separate
       ``reported_mult`` scaling factor applied to the remaining records.

    After redistribution the total adjusted weight is verified to equal the
    original in-use weight; a ``RuntimeError`` is raised if they differ.

    Args:
        vius: VIUS microdata DataFrame, one row per survey respondent.
        params: Pipeline parameters dict with keys:

            - ``home_base_corresp`` (dict): ``values``, ``id_columns``, used
              to build a correspondence table mapping VIUS home-base codes to
              internal category labels.
            - ``cab_type_corresp`` (dict): same structure for cab-type codes.
            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names.
            - ``home_source_col`` (str): raw column containing home-base code.
            - ``cab_source_col`` (str): raw column containing cab-type code.
            - ``totals_col`` (str): survey weight column name.
            - ``id_cols`` (dict[str, str]): mapping from logical name to
              column name for the final grouping dimensions (``region``,
              ``weight_class``, ``cab_type``).

    Returns:
        A ``pd.DataFrame`` with one row per (region, weight_class, cab_type)
        combination and a single column for the adjusted fleet-size total.

    Raises:
        RuntimeError: If the sum of adjusted survey weights does not match
            the sum of original in-use survey weights.
    """
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
