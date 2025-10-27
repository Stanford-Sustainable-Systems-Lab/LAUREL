from typing import Literal

import pandas as pd


class LocGroupingUniformityEvaluator:
    """Evaluate dwelling uniformity once and serve multiple summaries."""

    def __init__(self, dwell_locs: pd.Series, loc_groups: pd.Series) -> None:
        self.cutoff = None
        self.loc_col = dwell_locs.name or "location_id"
        self.grp_col = loc_groups.name or "group_id"

        loc_obs = loc_groups.to_frame(name=self.grp_col)
        loc_counts = dwell_locs.value_counts()
        loc_obs["n_dwells_observed"] = (
            loc_obs.index.map(loc_counts).fillna(0).astype(int)
        )

        loc_obs["n_dwells_uniform"] = loc_obs.groupby(self.grp_col)[
            "n_dwells_observed"
        ].transform(lambda ser: ser.sum() / ser.size)
        loc_obs["obs_uniform_ratio"] = (
            loc_obs["n_dwells_observed"] / loc_obs["n_dwells_uniform"]
        )
        self.loc_obs = loc_obs

    def set_cutoff(self, cutoff: float) -> None:
        """Update the cutoff and refresh the exceedance mask."""
        self.cutoff = cutoff
        self.loc_obs["exceeds_cutoff"] = self.loc_obs["obs_uniform_ratio"] > cutoff

    def summary(
        self, kind: Literal["raw", "overall", "group"] = "overall"
    ) -> float | pd.Series | pd.DataFrame:
        if self.cutoff is None:
            raise ValueError("Cutoff must be set before summaries can be made.")
        if kind == "raw":
            return self.loc_obs["exceeds_cutoff"]
        if kind == "overall":
            return self.loc_obs["exceeds_cutoff"].sum() / len(self.loc_obs)
        if kind == "group":
            loc_exc = self.loc_obs.groupby(self.grp_col).agg(
                exceeds_cutoff=pd.NamedAgg(column="exceeds_cutoff", aggfunc="sum"),
                group_size=pd.NamedAgg(column="exceeds_cutoff", aggfunc="count"),
            )
            loc_exc["exceeds_cutoff_frac"] = (
                loc_exc["exceeds_cutoff"] / loc_exc["group_size"]
            )
            return loc_exc
        raise NotImplementedError("Summary type not yet implemented.")

    def summary_raw(self) -> pd.Series:
        return self.summary("raw")

    def summary_overall(self) -> float:
        return self.summary("overall")

    def summary_group(self) -> pd.DataFrame:
        return self.summary("group")
