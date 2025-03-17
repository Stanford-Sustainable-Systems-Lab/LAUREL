"""
This is a boilerplate pipeline 'prepare_totals'
generated using Kedro 0.19.11
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from megaPLuG.utils.params import build_df_from_dict


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

    # Remove from consideration remaining vehicles with unknown operating distance
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
    totals = totals.loc[:, params["keep_group_cols"] + [params["final_totals_col"]]]
    totals = totals.sort_values(params["keep_group_cols"])
    return totals


def build_mandates_by_group(
    mands: pd.DataFrame, mand_states: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Take the ACF mandates dataframe and expand it out to be merge-able with
    Ledna adoptions.
    """
    cor_pars = params["acf_groups"]
    corresp = build_df_from_dict(
        d=cor_pars["values"],
        id_cols=cor_pars["id_columns"],
        value_col=cor_pars["value_column"],
    )
    groups = corresp.merge(mands, how="left", on=cor_pars["value_column"])
    groups = groups.drop(columns=[cor_pars["value_column"]])
    groups = groups.set_index(cor_pars["id_columns"])

    years = params["frame_years"]
    dates = pd.RangeIndex(start=years["min"], stop=years["max"] + 1).to_series()
    dates.index.name = "year"
    dfs = {grp: dates for grp in groups.index.unique()}
    frame = pd.concat(dfs, axis=0, names=cor_pars["id_columns"]).reset_index()
    frame = frame.drop(columns=[0])

    frac_col = params["mandate_fraction_col"]
    interp = frame.merge(groups, how="left", on=cor_pars["id_columns"] + ["year"])
    interp[frac_col] = interp.groupby(cor_pars["id_columns"])[frac_col].transform(
        lambda s: s.interpolate(method="linear")
    )
    interp[frac_col] = interp[frac_col].fillna(0.0)

    mou_states = {state: interp for state in mand_states[params["mou_state_col"]]}
    fracs = pd.concat(
        mou_states, axis=0, names=[params["mou_state_col"], "to_drop"]
    ).reset_index()
    fracs = fracs.drop(columns=["to_drop"])
    fracs = fracs.rename(columns={v: k for k, v in params["col_renamer"].items()})
    fracs[params["weight_class_col"]] = fracs[params["weight_class_col"]].astype(str)
    return fracs


def concat_projections_with_mandates(
    adopts: pd.DataFrame, mands: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Concatenate onto the original adoption projections a new set with mandates.

    Throughout, we refer to the group as the combination of all variables which set the
    type of vehicle we are talking about **other than fuel type**.

    Also, an "mclass" refers to the mandate class. If a vehicle is in the mandate class,
    **and** the mandate share exceeds the projected share for those mandated vehicles,
    then they will get the mandate share applied to them. If a vehicle is not in the
    mandate class, then they will get 1 - mandate share applied to them.
    """
    # Prepare for grouping calculations
    group_cols = params["group_cols"]
    adopts = adopts.sort_values(group_cols + [params["fuel_type_col"]])
    adopts["is_zev"] = adopts[params["fuel_type_col"]].isin(params["zev_fuel_types"])

    # Get the adoption projection probability of a vehicle being in a mandate class
    # given that they are in a particular group
    tot_col = params["totals_col"]
    n_vehs_in_group = adopts.groupby(group_cols)[tot_col].sum().reset_index()
    n_vehs_in_group_mclass = (
        adopts.groupby(group_cols + ["is_zev"])[tot_col].sum().reset_index()
    )
    p_mclass_g_group = n_vehs_in_group_mclass.merge(
        n_vehs_in_group, how="left", on=group_cols, suffixes=("_group_mclass", "_group")
    )
    tot_col_group = f"{tot_col}_group"
    tot_col_group_mclass = f"{tot_col}_group_mclass"
    p_mclass_g_group["p_mclass_g_group"] = (
        p_mclass_g_group[tot_col_group_mclass] / p_mclass_g_group[tot_col_group]
    )

    # Apply the mandate override to groups where the mandate requires more ZEVs than
    # the forecast predicts.
    over_group_cols = group_cols.copy()
    over_group_cols.remove(params["scenario_col"])
    override = p_mclass_g_group.merge(mands, how="left", on=over_group_cols)
    override["mandate_override"] = (
        override[params["mandate_fraction_col"]] > override["p_mclass_g_group"]
    )
    group_override = override.groupby(group_cols)["mandate_override"].any()
    override = override.drop(columns=["mandate_override"])
    override = override.merge(group_override, how="left", on=group_cols)
    override["mandate_override"] = override["mandate_override"].fillna(False)
    ovr = override["mandate_override"]
    is_zev = override["is_zev"]
    mand_col = "mandate_mclass_g_group"
    override.loc[ovr & is_zev, mand_col] = override.loc[
        ovr & is_zev, params["mandate_fraction_col"]
    ]
    override.loc[ovr & ~is_zev, mand_col] = (
        1 - override.loc[ovr & ~is_zev, params["mandate_fraction_col"]]
    )
    override.loc[~ovr, mand_col] = override.loc[~ovr, "p_mclass_g_group"]

    scen_new = adopts.merge(override, how="left", on=group_cols + ["is_zev"])

    # Get the probability of a particular technology within a group and mandate class
    # from the adoption projections. If the adoption projections aren't complete, then
    # assume a default technology within each mandate group.
    scen_new["p_tech_g_group_mclass"] = (
        scen_new[tot_col] / scen_new[tot_col_group_mclass]
    )
    zero_denom = (
        scen_new[tot_col_group_mclass] == 0
    )  # Cases where Ledna predicted no adoption at all in this emissions class
    default_fuel = scen_new[params["fuel_type_col"]].isin(params["default_fuel_types"])
    scen_new.loc[zero_denom & default_fuel, "p_tech_g_group_mclass"] = 1.0
    scen_new.loc[zero_denom & ~default_fuel, "p_tech_g_group_mclass"] = 0.0

    # Calculate the new target numbers of vehicles
    ovr = scen_new["mandate_override"]
    tot_col_new = f"{tot_col}_new"
    scen_new.loc[~ovr, tot_col_new] = scen_new.loc[~ovr, tot_col]
    scen_new.loc[ovr, tot_col_new] = (
        scen_new.loc[ovr, tot_col_group]
        * scen_new.loc[ovr, "mandate_mclass_g_group"]
        * scen_new.loc[ovr, "p_tech_g_group_mclass"]
    ).astype(pd.Int64Dtype())
    scen_new = scen_new.sort_values(group_cols + [params["fuel_type_col"]])

    # Clean up the new targets
    keep_cols = group_cols + [params["fuel_type_col"]] + [tot_col_new]
    drop_cols = set(scen_new.columns).difference(keep_cols)
    out = scen_new.drop(columns=drop_cols)
    out = out.rename(columns={tot_col_new: tot_col})

    # Clean up the old targets
    out_adopts = adopts.drop(columns=["is_zev"])

    # Concatenate the new and old targets together
    jnt = pd.concat(
        [out_adopts, out],
        axis=0,
        keys=[False, True],
        names=[params["mandate_active_col"]],
    )
    jnt = jnt.droplevel(level=1, axis=0)
    jnt = jnt.reset_index()
    return jnt
