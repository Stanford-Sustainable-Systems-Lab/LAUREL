"""
This is a boilerplate pipeline 'prepare_totals'
generated using Kedro 0.19.11
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def prepare_for_merging(vius: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Prepare the VIUS microdata for merging of groups."""
    scaler = vius.rename(columns={v: k for k, v in params["col_renamer"].items()})

    dist_cols = [
        col for col in scaler.columns if col.startswith(params["dist_bin_col_prefix"])
    ]
    all_dist_na = scaler[dist_cols].isna().all(axis=1)
    scaler[params["primary_dist_col"]] = "NA"
    scaler.loc[~all_dist_na, params["primary_dist_col"]] = scaler.loc[
        ~all_dist_na, dist_cols
    ].idxmax(axis=1, skipna=True)
    return scaler


def aggregate_vius_totals(vius: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Aggregate VIUS totals, and prepare for that aggregation."""
    enc = OneHotEncoder(sparse_output=False)
    ohot = enc.fit_transform(vius.loc[:, [params["home_base_code_col"]]])
    ohot = pd.DataFrame(ohot, columns=enc.categories_[0], dtype=bool)

    reg_col = params["home_base_region_col"]
    vius.loc[ohot["Home Base in Register State"], reg_col] = vius.loc[
        ohot["Home Base in Register State"], params["region_source_col"]
    ]
    vius.loc[ohot["No Home Base"], reg_col] = "No Home Base"
    vius.loc[ohot["Home Base not in Register State"], reg_col] = (
        "Home Base not in Register State"
    )
    vius.loc[vius[reg_col].isna(), params["weight_col"]] = 0.0

    # Set the operating distance of "No Home Base" vehicles to a default
    vius.loc[ohot["No Home Base"], params["op_dist_col"]] = params["fill_op_dist"]
    vius.loc[vius[params["op_dist_col"]] == "NA", params["op_dist_col"]] = pd.NA
    vius.loc[vius[params["op_dist_col"]].isna(), params["weight_col"]] = 0.0

    # Spread weight from unknown home base state to known home base state proportionally
    vius["wgt_specific"] = (
        vius[params["weight_col"]] * ohot["Home Base in Register State"]
    )
    vius["wgt_home_base"] = vius[params["weight_col"]] * (
        ohot["Home Base in Register State"] | ohot["Home Base not in Register State"]
    )

    grped = vius.groupby(params["spread_condition_cols"], observed=True).agg(
        wgt_specific=pd.NamedAgg("wgt_specific", "sum"),
        wgt_home_base=pd.NamedAgg("wgt_home_base", "sum"),
    )
    grped["p_home_base_known_g_has_home_base"] = (
        grped["wgt_specific"] / grped["wgt_home_base"]
    )
    grped["specific_mult"] = 1 / grped["p_home_base_known_g_has_home_base"]
    mrg = grped.loc[:, "specific_mult"].reset_index()

    vius_adj = vius.merge(mrg, how="left", on=params["spread_condition_cols"])
    vius_adj.loc[~ohot["Home Base in Register State"], "specific_mult"] = 1.0
    vius_adj[params["weight_col"]] = (
        vius_adj[params["weight_col"]] * vius_adj["specific_mult"]
    )
    drop_idx = vius_adj.loc[ohot["Home Base not in Register State"]].index
    drop_cols = ["wgt_specific", "wgt_home_base", "specific_mult"]
    vius_adj = vius_adj.drop(index=drop_idx, columns=drop_cols)

    # Aggregate into a disaggregation crosswalk for national adoption forecasts
    jnt = (
        vius_adj.groupby(params["group_cols"])[params["weight_col"]].sum().reset_index()
    )
    jnt["given_total"] = jnt.groupby(params["disagg_condition_cols"])[
        params["weight_col"]
    ].transform(lambda x: x.sum())
    jnt[params["out_prob_col"]] = jnt[params["weight_col"]] / jnt["given_total"]
    jnt = jnt.drop(columns=[params["weight_col"], "given_total"])
    return jnt


def aggregate_adoption_forecast_totals(
    adopts: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Aggregate the adoption forecast totals to prepare for merging."""
    adpt = adopts.rename(columns={v: k for k, v in params["col_renamer"].items()})
    wgt_col = params["weight_col"]
    adpt = adpt.groupby(params["group_cols"])[wgt_col].sum().reset_index()
    adpt[wgt_col] = adpt[wgt_col] * params["vehicle_stock_mult"]
    return adpt


def create_disaggregated_adoption(
    adopts: pd.DataFrame, disagg: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Merge a disaggregation factor dataframe onto an adoption forecast dataframe."""
    totals = adopts.merge(disagg, how="left", on=params["merge_cols"])
    totals[params["final_totals_col"]] = (
        (totals[params["orig_totals_col"]] * totals[params["disagg_factor_col"]])
        .round(0)
        .astype(pd.Int64Dtype())
    )
    return totals
