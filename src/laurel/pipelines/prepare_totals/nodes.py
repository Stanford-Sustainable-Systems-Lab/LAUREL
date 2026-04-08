"""Kedro pipeline nodes for the ``prepare_totals`` pipeline (Model Module 1 — Select SoWs).

Prepares the vehicle-count denominators and adoption-projection tables that
the ``evaluate_impacts`` pipeline uses to scale telematics observations to
realistic fleet-level charging loads.  This pipeline implements the first
half of Model Module 1 (Select States of the World): constructing the
disaggregation crosswalk that maps national NREL Ledna adoption forecasts
down to the (region × operating-distance class × weight class) strata used
by the model.  An optional ACF (Advanced Clean Fleets) mandate variant
overrides projections wherever the regulatory minimum exceeds the forecast.

Pipeline overview
-----------------
1. **prepare_for_merging** — Renames VIUS microdata columns and derives each
   respondent's primary operating-distance class from a set of distance-bin
   indicator columns.
2. **aggregate_vius_totals** — One-hot encodes the home-base-code variable,
   redistributes survey weight from out-of-state or unknown home-base
   respondents, and aggregates into a conditional probability table
   ``P(region, op_dist | weight_class)`` used as a disaggregation factor.
3. **aggregate_adoption_forecast_totals** — Renames and aggregates the raw
   NREL Ledna vehicle-count projections to the required group level, applying
   a vehicle-stock multiplier to convert units if necessary.
4. **create_disaggregated_adoption** — Merges the VIUS disaggregation factors
   onto the national adoption totals to produce stratum-level vehicle counts.
5. **build_mandates_by_group** — Expands the sparse ACF mandate schedule into
   a dense (year × operating-distance class × weight class × state) table,
   interpolating linearly between known mandate fractions and broadcasting
   each mandate to the set of MOU-signatory states.
6. **concat_projections_with_mandates** — Appends a mandate-adjusted copy of
   the adoption projections in which ZEV fractions that fall below the
   regulatory minimum are raised to the mandate level, while non-ZEV fractions
   are correspondingly reduced.

Key design decisions
--------------------
- **Proportional weight redistribution**: Vehicles with "Home Base not in
  Register State" cannot be attributed to a known state, so their weight is
  redistributed to same-stratum "Home Base in Register State" respondents.
  This preserves the marginal totals while making every vehicle attributable.
- **Zero-denominator default technology**: When Ledna projects zero adoption
  in an entire emissions class, the model defaults to a single representative
  fuel type (``default_fuel_types``) with probability 1.0 rather than leaving
  the conditional probability undefined.
- **Mandate override threshold**: The mandate override is only applied when
  the mandate fraction *exceeds* the forecast ZEV fraction for a group; if the
  market forecast already meets or exceeds the mandate, the forecast is left
  unchanged.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.

NREL. (2023). Ledna: Light- and Medium-Duty Electric Vehicle Adoption Model.
California Air Resources Board. Advanced Clean Fleets regulation.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from laurel.utils.params import build_df_from_dict


def prepare_for_merging(vius: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Rename VIUS columns and derive each respondent's primary operating-distance class.

    The VIUS encodes operating distance as a set of binary indicator columns
    (one per distance bin).  This function identifies the bin with the
    highest value for each respondent to assign a single ``primary_dist_col``
    label.  Respondents with all-NA distance bins are assigned a sentinel
    value of ``"NA"`` so they can be excluded downstream.

    Args:
        vius: Raw VIUS microdata DataFrame as loaded by the ``preprocess``
            pipeline.
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw column names
              to internal names.
            - ``dist_bin_col_prefix`` (str): prefix shared by all operating-
              distance indicator columns (used to identify them by name).
            - ``primary_dist_col`` (str): name of the output column for the
              primary operating-distance label.

    Returns:
        The input DataFrame with columns renamed and a new ``primary_dist_col``
        column added.
    """
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
    """Build a disaggregation crosswalk from VIUS survey weights.

    Constructs the conditional probability table
    ``P(region, op_dist | weight_class)`` that is later used to disaggregate
    national adoption-forecast totals into the (region, operating-distance
    class, weight class) strata required by the model.

    The procedure handles three home-base categories:

    - **"Home Base in Register State"**: Retained and scaled by
      ``1 / P(known_home_base | has_home_base)`` to absorb the weight of
      out-of-state records.
    - **"Home Base not in Register State"**: Dropped after contributing to the
      scaling denominator.
    - **"No Home Base"**: Retained but assigned a default operating distance.
    - Respondents with unknown operating distance have their weight zeroed out.

    The final groupby aggregation over ``group_cols`` normalises by the
    conditional total to produce probabilities.

    Args:
        vius: VIUS microdata with ``primary_dist_col`` already assigned (output
            of ``prepare_for_merging``).
        params: Pipeline parameters dict with keys:

            - ``home_base_code_col`` (str): column containing the home-base
              category label.
            - ``home_base_region_col`` (str): output column for the region
              label derived from the home-base category.
            - ``region_source_col`` (str): raw column providing the state name
              for "Home Base in Register State" records.
            - ``weight_col`` (str): survey weight column.
            - ``op_dist_col`` (str): operating-distance class column.
            - ``fill_op_dist`` (str): default operating-distance label for
              "No Home Base" vehicles.
            - ``spread_condition_cols`` (list[str]): columns defining strata
              within which the home-base-known probability is computed.
            - ``group_cols`` (list[str]): columns to group by for the final
              aggregation.
            - ``disagg_condition_cols`` (list[str]): conditioning columns for
              the normalisation denominator.
            - ``out_prob_col`` (str): name of the output probability column.

    Returns:
        A ``pd.DataFrame`` with one row per (group_cols) combination and a
        probability column ``out_prob_col`` summing to 1.0 within each
        ``disagg_condition_cols`` stratum.
    """
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
    """Aggregate and unit-scale the NREL Ledna adoption-forecast vehicle counts.

    Renames columns, groups to the required dimensionality, and multiplies
    by a stock scalar (e.g., to convert from thousands of vehicles to
    individual vehicles).

    Args:
        adopts: Raw Ledna adoption-forecast DataFrame.
        params: Pipeline parameters dict with keys:

            - ``col_renamer`` (dict[str, str]): mapping from raw to internal
              column names.
            - ``group_cols`` (list[str]): columns to group by.
            - ``weight_col`` (str): vehicle-count column.
            - ``vehicle_stock_mult`` (float): scalar multiplier applied to
              the aggregated totals (e.g., 1000 if Ledna reports in thousands).

    Returns:
        A ``pd.DataFrame`` with one row per ``group_cols`` combination and
        a scaled vehicle-count column.
    """
    adpt = adopts.rename(columns={v: k for k, v in params["col_renamer"].items()})
    wgt_col = params["weight_col"]
    adpt = adpt.groupby(params["group_cols"])[wgt_col].sum().reset_index()
    adpt[wgt_col] = adpt[wgt_col] * params["vehicle_stock_mult"]
    return adpt


def create_disaggregated_adoption(
    adopts: pd.DataFrame, disagg: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Disaggregate national adoption totals to stratum-level vehicle counts.

    Merges the VIUS-derived conditional probability table (``disagg``) onto
    the aggregated adoption forecast (``adopts``) and multiplies to produce
    vehicle counts broken out by region and operating-distance class.

    Args:
        adopts: Aggregated adoption totals (output of
            ``aggregate_adoption_forecast_totals``).
        disagg: Disaggregation crosswalk with a probability column (output of
            ``aggregate_vius_totals``).
        params: Pipeline parameters dict with keys:

            - ``merge_cols`` (list[str]): columns to join on.
            - ``orig_totals_col`` (str): vehicle-count column in ``adopts``.
            - ``disagg_factor_col`` (str): probability column in ``disagg``.
            - ``final_totals_col`` (str): name of the output vehicle-count
              column (rounded to the nearest integer).
            - ``keep_group_cols`` (list[str]): columns to retain in the
              output.

    Returns:
        A ``pd.DataFrame`` with integer vehicle counts per stratum, sorted
        by ``keep_group_cols``.
    """
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
    """Expand the sparse ACF mandate schedule into a dense, merge-ready table.

    The raw ACF mandate data provides ZEV fraction requirements at a small
    number of (year, vehicle-class) combinations.  This function:

    1. Maps ACF vehicle classes to the model's operating-distance/weight-class
       groups via a correspondence table.
    2. Creates a full (group × year) grid spanning ``frame_years``.
    3. Linearly interpolates mandate fractions between known anchor years;
       years before the first anchor are filled with 0.0.
    4. Broadcasts the result across all MOU-signatory states in
       ``mand_states``.

    Args:
        mands: Sparse ACF mandate DataFrame with (year, class, fraction) rows.
        mand_states: DataFrame of MOU-signatory state identifiers.
        params: Pipeline parameters dict with keys:

            - ``acf_groups`` (dict): ``values`` (correspondence mapping),
              ``id_columns`` (list), ``value_column`` (str) — defines the
              mapping from ACF vehicle class to model group.
            - ``frame_years`` (dict): ``min`` and ``max`` year for the output
              grid.
            - ``mandate_fraction_col`` (str): column name for the ZEV fraction.
            - ``mou_state_col`` (str): column in ``mand_states`` containing
              state identifiers.
            - ``col_renamer`` (dict[str, str]): mapping to harmonise column
              names with the adoption-forecast tables.
            - ``weight_class_col`` (str): weight-class column to cast to str.

    Returns:
        A ``pd.DataFrame`` with one row per (state, group, year) combination
        and a ``mandate_fraction_col`` column suitable for merging with
        adoption-forecast tables.
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
    """Append a mandate-adjusted copy of the adoption projections to the original forecasts.

    The group is defined as the combination of all variables that characterise a
    vehicle type *other than* fuel type (e.g., weight class, operating distance,
    region, year, scenario).  The *mandate class* (``mclass``) is the ZEV
    designation: a vehicle is in the mandate class if and only if its fuel type
    is in ``zev_fuel_types``.

    The override logic is:

    1. Compute ``P(mclass | group)`` from the adoption forecast.
    2. Merge the mandate fraction for each group from ``mands``.
    3. If the mandate fraction exceeds ``P(mclass | group)``, mark the group
       as ``mandate_override = True``.
    4. For overridden groups, replace ZEV counts with
       ``N_group × mandate_fraction × P(tech | group, mclass)`` and non-ZEV
       counts with ``N_group × (1 − mandate_fraction) × P(tech | group, ~mclass)``.
    5. When Ledna projected zero ZEVs in a group, default to a single
       representative technology (``default_fuel_types``) with probability 1.

    The original and mandate-adjusted projections are stacked with a boolean
    index level indicating whether the mandate is active.

    Args:
        adopts: Disaggregated adoption-forecast totals.
        mands: Dense mandate schedule (output of ``build_mandates_by_group``).
        params: Pipeline parameters dict with keys:

            - ``group_cols`` (list[str]): columns that define a group (all
              dimensions except fuel type).
            - ``fuel_type_col`` (str): column containing the fuel-type label.
            - ``zev_fuel_types`` (list[str]): fuel-type labels counted as ZEVs.
            - ``totals_col`` (str): vehicle-count column.
            - ``scenario_col`` (str): scenario identifier (excluded from the
              mandate merge join to apply mandates across all scenarios).
            - ``mandate_fraction_col`` (str): ZEV fraction column from
              ``mands``.
            - ``default_fuel_types`` (list[str]): fallback fuel type when
              Ledna projected zero ZEVs.
            - ``mandate_active_col`` (str): name of the boolean index level
              added to the output.

    Returns:
        A ``pd.DataFrame`` with a boolean index level ``mandate_active_col``
        (``False`` = original forecast, ``True`` = mandate-adjusted forecast)
        and the same columns as ``adopts``.
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
