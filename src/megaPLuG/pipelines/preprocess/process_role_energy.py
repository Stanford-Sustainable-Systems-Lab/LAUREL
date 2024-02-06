# # RoL-E Charging Trajectory Inventory
#
# *by Fletcher Passow*
#
# February 2023

from itertools import product

import numpy as np
import pandas as pd

from .events import get_events


def label_charging_sessions(energy_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Clean RoLE energy data then label charging sessions."""
    energy_df["timestamp"] = pd.to_datetime(
        energy_df["timestamp"], utc=True, format="mixed"
    )
    energy_df = energy_df.drop(columns=["id", "geotab_id", "geotab_import_id"])

    # ### Drop duplicated rows
    # These were found in truck 11, but may be in other trucks too!
    dup_idx = energy_df.loc[energy_df.duplicated()].index
    energy_df = energy_df.drop(index=dup_idx)

    # ### Drop Truck 8, which only has one record every day instead of more frequently
    # This data may still be useful, since it is still a truck-day, but it isn't
    # any good for charging sessions.
    truck_drop_idx = energy_df.loc[
        energy_df["truck_id"].isin(params["truck_ids_to_drop"])
    ].index
    energy_df = energy_df.drop(index=truck_drop_idx)

    # ### Calculate incremental energy from cumulative energy
    energy_df = energy_df.sort_values(["truck_id", "timestamp"])
    energy_df["kwh_diff"] = energy_df.groupby(["truck_id", "reading_type"])[
        "kwh"
    ].diff()

    # ### Removing spurious incremental energy values
    # Since all of the energy values are defined in such a way as to always be
    # positive, we assume that the negative values reported here are spurious.
    # To correct for these values, we clip them to zero.
    energy_df["kwh_diff"] = energy_df["kwh_diff"].clip(lower=0.0)

    # ### Pivot energy readings wide
    # This will let us get a more easily understandable charging session definition. We can then compare this definition to RoL-E's original definition.
    pivot_df = energy_df.pivot(
        index=["truck_id", "timestamp"], columns=["reading_type"], values="kwh_diff"
    ).sort_index()
    pivot_df = pivot_df.replace(to_replace=np.NaN, value=0.0)

    # Calculate time differences
    pivot_df = pivot_df.sort_index().reset_index(["truck_id", "timestamp"])
    pivot_df["time_diff"] = pivot_df.groupby("truck_id")["timestamp"].diff()
    pivot_df["time_diff_sec"] = pivot_df["time_diff"].dt.total_seconds()

    # Calculate charger powers
    pivot_df["charger_energy"] = (
        pivot_df["energy_from_ac_charger"] + pivot_df["energy_from_dc_charger"]
    )
    pivot_df["is_active"] = (
        pivot_df["driving_energy"] + pivot_df["energy_regen"] + pivot_df["idle_energy"]
    ) > 0
    pivot_df["charger_power_kw"] = (
        pivot_df["charger_energy"] * 3600 / pivot_df["time_diff_sec"]
    )  # Convert to kJ before dividing to make kW final unit
    replace_series = np.logical_and(
        pivot_df["charger_power_kw"].isna(), pivot_df["charger_energy"] == 0
    )  # Deal with divide by zero errors where the numerator is also zero
    pivot_df.loc[replace_series, "charger_power_kw"] = 0.0

    # Clip observations with impossibly high charging rates, which all occur in
    # trucks 6 and 7. Truck 7 seems to have a problem with many observations being
    # too high in the same way. Truck 6 has just one or two observations that are
    # wildly high.

    # Observe how there are tons of observations at about 37 kW, then it shoots off rapidly
    # pivot_df.query('truck_id == 7 & charger_power_kw > 0').sort_values('charger_power_kw').iloc[-100:-50]
    TRUCK_ID = 7
    max_pow = params["max_possible_charger_power_kw"]
    clip_selector = np.logical_and(
        pivot_df["truck_id"] == TRUCK_ID, pivot_df["charger_power_kw"] > max_pow
    )
    pivot_df.loc[clip_selector, "charger_power_kw"] = pivot_df.loc[
        clip_selector, "charger_power_kw"
    ].clip(upper=max_pow)
    pivot_df.loc[clip_selector, "charger_energy"] = (
        pivot_df.loc[clip_selector, "charger_power_kw"]
        * pivot_df.loc[clip_selector, "time_diff_sec"]
        / 3600
    )
    pivot_df.loc[clip_selector, "energy_from_ac_charger"] = pivot_df.loc[
        clip_selector, "charger_energy"
    ]
    # pivot_df.query('truck_id == 7 & charger_power_kw > 0').sort_values('charger_power_kw').iloc[-100:-50]

    # Now re-examining the observations that remain. Truck 6 definitely has some
    # extremely high values. Charging powers of around 100kW from a DC charger seem
    # acceptable, but I assume that anything over 200kW is erroneous. Some batteries
    # on the road now can charge at that rate, but we should see that rate a little
    # more frequently in this dataset if it was real.

    # pivot_df.sort_values("charger_power_kw").tail(20)

    # Based on examination of SoC data, it is clear that the charging session on
    # 2021-08-10 at 20:53:58 is spurious. The SoC did not change at all during that
    # time. The same is true of the charging session on 2021-08-11 at 17:40:15.
    # Therefore, I assume that the SoC is correct and I zero out both of these high
    # energy observations.

    TRUCKS_TO_ZERO = [6]
    TIMES_TO_ZERO = ["2021-08-10 20:53:58.063000+00:00", "2021-08-11 17:40:15+00:00"]
    COLS_TO_ZERO = ["energy_from_ac_charger", "charger_energy", "charger_power_kw"]
    pivot_df = pivot_df.set_index(["truck_id", "timestamp"])
    pivot_df.loc[product(TRUCKS_TO_ZERO, TIMES_TO_ZERO), COLS_TO_ZERO] = 0

    # TODO: Next, dealing with extremely large `driving_energy`, `energy_regen`, and
    # `idle_energy` observations from truck 7 at 2021-08-31 21:12:23.880000+00:00.
    # This observation has a very long time elapsed before it. It also has lots of
    # normal driving both immediately before it and immediately after it. This also
    # falls in the normal driving time for truck 7 (operated by Purolator). Based on
    # this information, it seems probable that there actually was driving during
    # that time, but that there was probably not charging. Therefore, we can say
    # that an upper bound to net energy consumed would be the total capacity of the
    # battery.

    ### ===================== Define charging events ===========================
    ### >>>>>>>>>>>>>>>>>>>>> First method >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # The first charging event definition assumes that charging events are
    # contiguous stretches of charging energy being applied to the truck. Gaps of
    # time with zero charging power are allowed, as long as the gap is shorter than
    # a threshold defined by `params["max_charge_gap_secs"]`.
    test_df = pivot_df.reset_index()
    pivot_df["direct_sess_id"] = get_events(
        obs_ser=test_df["charger_energy"],
        time_ser=test_df["timestamp"],
        grp_ser=test_df["truck_id"],
        max_time_elapsed=params["max_charge_gap_secs"],
    )

    ### >>>>>>>>>>>>>>>>>>>>> Second method >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # This second charging event definition assumes that any charging sessions
    # that occur within the same "non-driving" event are a single contiguous
    # charging event. This lets us deal with very long dormant periods in a
    # principled way.

    # First, build "non-driving" events with zero time elapsed tolerance. This
    # assumes that we have continuous data collection, and that whenever we're
    # getting no data it's because the truck was turned off and idle. In the
    # NACFE RoL-E final report, they do not share any big data quality issues.
    event_df = pivot_df.reset_index()
    pivot_df["no_driving_sess_id"] = get_events(
        obs_ser=event_df["is_active"],
        time_ser=event_df["timestamp"],
        grp_ser=event_df["truck_id"],
        obs_in_event=lambda x: not x,
        max_time_elapsed=0,
    )

    # Now, selecting charging sessions by using "non-driving" events with any
    # charging within them.
    sess_energy_ser = pivot_df.groupby("no_driving_sess_id")["charger_energy"].sum()
    has_no_energy_idx = sess_energy_ser.loc[sess_energy_ser == 0].index
    sess_energy_ser = sess_energy_ser.drop(index=has_no_energy_idx)
    has_energy_idx = sess_energy_ser.index

    event_df = pivot_df.reset_index()
    pivot_df["nondrive_sess_id"] = get_events(
        obs_ser=event_df["no_driving_sess_id"],
        time_ser=event_df["timestamp"],
        grp_ser=event_df["truck_id"],
        obs_in_event=lambda x: x.isin(has_energy_idx),
        max_time_elapsed=0,
    )
    pivot_df = pivot_df.drop(columns=["no_driving_sess_id"])

    # Now, clipping the no-energy periods at the beginning and end of charging
    # sessions. I achieve this using forward and backward cumulative sums across
    # the events.
    pivot_df["sess_cumsum_forward"] = pivot_df.groupby("nondrive_sess_id")[
        "charger_energy"
    ].cumsum()
    pivot_df = pivot_df.sort_index(ascending=False)
    pivot_df["sess_cumsum_backward"] = pivot_df.groupby("nondrive_sess_id")[
        "charger_energy"
    ].cumsum()
    pivot_df = pivot_df.sort_index()

    non_tail_ser = np.logical_and(
        pivot_df["sess_cumsum_forward"] > 0, pivot_df["sess_cumsum_backward"] > 0
    )
    pivot_df["nondrive_sess_id"] = pivot_df["nondrive_sess_id"] * non_tail_ser
    pivot_df = pivot_df.drop(columns=["sess_cumsum_forward", "sess_cumsum_backward"])

    ### ===================== Write outputs ====================================
    # Write out interim dataframe
    pivot_df = pivot_df.drop(columns=["is_active", "time_diff"])
    return pivot_df
