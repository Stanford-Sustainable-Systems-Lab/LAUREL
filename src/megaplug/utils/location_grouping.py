"""Evaluator for dwell-count uniformity within location groups (TAZ clusters).

When hexagons are grouped into freight-activity clusters for the K-Means
classification step, an ideal clustering assigns roughly equal numbers of
observed dwells to every location within a group.  This module provides
:class:`LocGroupingUniformityEvaluator`, which computes an ``obs/uniform``
ratio for each location (how many times more dwells were observed at this
location than expected under a uniform distribution within the group) and
exposes summary statistics at multiple levels of aggregation.

This evaluator is used in exploratory notebooks to validate that the K-Means
clustering produces geographically coherent groups rather than concentrating
all observed activity in a few locations per cluster.
"""

from typing import Literal

import pandas as pd


class LocGroupingUniformityEvaluator:
    """Evaluate dwell-count uniformity within location groups and serve multiple summaries.

    On construction, computes for every location:

    - ``n_dwells_observed``: actual dwell count at this location.
    - ``n_dwells_uniform``: expected count if dwells were spread evenly across
      all locations in the same group.
    - ``obs_uniform_ratio``: the ratio of the two.

    After setting a cutoff via :meth:`set_cutoff`, the ``summary`` method
    returns statistics at three granularities: raw boolean mask, scalar
    overall fraction, or per-group breakdown.

    Args (constructor):
        dwell_locs: Series of location IDs, one entry per dwell observation.
        loc_groups: Series indexed by location ID, values are group labels.
    """

    def __init__(self, dwell_locs: pd.Series, loc_groups: pd.Series) -> None:
        self.cutoff = None
        self.loc_col = dwell_locs.name or "location_id"
        self.grp_col = loc_groups.name or "group_id"

        loc_obs = loc_groups.to_frame(name=self.grp_col)
        loc_counts = dwell_locs.value_counts()
        loc_obs["n_dwells_observed"] = (
            loc_obs.index.map(loc_counts).fillna(0).astype(int)
        )

        loc_obs["n_dwells_uniform"] = loc_obs.groupby(self.grp_col, observed=True)[
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
        """Return a uniformity summary at the requested aggregation level.

        Args:
            kind: One of:

                - ``"raw"`` — boolean Series, one entry per location, indicating
                  whether its ``obs_uniform_ratio`` exceeds the cutoff.
                - ``"overall"`` — scalar fraction of locations exceeding the
                  cutoff across the whole dataset.
                - ``"group"`` — per-group DataFrame with columns
                  ``exceeds_cutoff``, ``group_size``, ``uniform_dwell_rate``,
                  ``group_frac``, and ``exceeds_cutoff_frac``.

        Returns:
            Summary at the requested level.

        Raises:
            ValueError: If :meth:`set_cutoff` has not been called yet.
            NotImplementedError: If ``kind`` is not one of the three supported values.
        """
        if self.cutoff is None:
            raise ValueError("Cutoff must be set before summaries can be made.")
        if kind == "raw":
            return self.loc_obs["exceeds_cutoff"]
        if kind == "overall":
            return self.loc_obs["exceeds_cutoff"].sum() / len(self.loc_obs)
        if kind == "group":
            loc_exc = self.loc_obs.groupby(self.grp_col, observed=True).agg(
                exceeds_cutoff=pd.NamedAgg(column="exceeds_cutoff", aggfunc="sum"),
                group_size=pd.NamedAgg(column="exceeds_cutoff", aggfunc="count"),
                uniform_dwell_rate=pd.NamedAgg(
                    column="n_dwells_uniform", aggfunc="first"
                ),
            )
            loc_exc["group_frac"] = loc_exc["group_size"] / loc_exc["group_size"].sum()
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
