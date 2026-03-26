"""Plotting utilities for temporal load profiles and quantile bands.

Provides two components for visualising charging load profiles in reporting
notebooks:

- :class:`ProfileDensifier` — fills missing time-steps in a sparse profile
  DataFrame so that ``matplotlib``/``seaborn`` plots render without gaps.
- :func:`plot_quantile_bands` — overlays shaded confidence bands on a
  ``seaborn.FacetGrid`` for a set of quantile columns symmetric around the
  median.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from megaplug.models.group_times import AdaptiveTimeGrouper

MEDIAN = 0.5


class ProfileDensifier:
    """Class for densifying sparse temporal profiles using time groupers.

    Provides methods to create reusable time groupers and densify sparse
    DataFrames by filling missing time points with specified values.
    """

    def __init__(self, time_col: str, freq: str, dur: str = "1d"):
        """Initialize ProfileDensifier with time configuration.

        Args:
            time_col: Name of the column containing time/datetime values.
            freq: Frequency string for time series generation (e.g., "15min", "1h").
            dur: Duration string defining the total time span (default: "1d").
        """
        self.time_col = time_col
        self.freq = freq
        self.dur = dur
        self.grouper = None

    def create_grouper(self) -> AdaptiveTimeGrouper:
        """Create and return configured AdaptiveTimeGrouper for external use.

        Returns:
            Configured AdaptiveTimeGrouper instance that can be used independently.
        """
        if self.grouper is None:
            self.grouper = AdaptiveTimeGrouper(
                time_col=self.time_col,
                tz_col="tz",
                start_time=pd.Timestamp(0),
                end_time=pd.Timestamp(0)
                + pd.Timedelta(self.dur)
                - pd.Timedelta(self.freq),
                possible_tzs=["local_time"],
                freq=self.freq,
            )
        return self.grouper

    def densify(
        self,
        sparse: pd.DataFrame,
        grp_cols: list[object],
        value_cols: list[object],
        fill_value=0.0,
    ) -> pd.DataFrame:
        """Densify sparse profiles by filling missing time points.

        Args:
            sparse: Sparse DataFrame containing observed data with missing time points.
            grp_cols: List of column names to group by for densification.
            value_cols: List of column names containing values to be preserved.
            fill_value: Value to use for filling missing time points (default: 0.0).

        Returns:
            Complete DataFrame with densified profiles indexed by grp_cols + [time_col].
        """
        grouper = self.create_grouper()

        # Build all possible time combinations
        all_classes = grouper.get_possible_obs_counts().reset_index()
        all_classes = all_classes.drop(columns=["tz", grouper.count_col])
        all_classes[self.time_col] = all_classes[self.time_col].dt.tz_localize(None)

        # Apply time combinations to all groups
        components = []
        for col in grp_cols:
            if col in sparse.columns:
                vals = sparse[col]
            elif col in sparse.index.names:
                vals = sparse.index.get_level_values(col)
            else:
                raise RuntimeError(
                    f"Column {col} not found in the sparse dataframe columns or index names."
                )
            components.append(vals.unique())

        components.append(all_classes[self.time_col])
        idx_cols = grp_cols + [self.time_col]
        sub_idx = pd.MultiIndex.from_product(components, names=idx_cols)
        sub_frame = sub_idx.to_frame(index=False)

        # Merge and fill missing values
        mrg = sparse.reset_index().loc[:, idx_cols + value_cols]
        profs_frame = sub_frame.merge(mrg, how="left", on=idx_cols)
        profs_frame = profs_frame.set_index(idx_cols)
        profs_frame = profs_frame.fillna(fill_value)
        return profs_frame


def get_band_bounds(qtls: list[float] | pd.Index) -> list[tuple[float, float]]:
    """Pair quantiles symmetrically around the median for band plotting.

    Given a list of quantile levels that includes 0.5 and is of odd length,
    returns pairs ``(low, high)`` ordered from the outermost band inward, ready
    for :func:`plot_quantile_bands`.

    Args:
        qtls: Sorted list (or ``pd.Index``) of quantile levels.  Must contain
            the median (0.5) and have an odd number of entries.

    Returns:
        List of ``(lower_quantile, upper_quantile)`` tuples, outermost first.

    Raises:
        RuntimeError: If the median is absent or the length is even.
    """
    if isinstance(qtls, pd.Index):
        qtls = qtls.tolist()
    if MEDIAN not in qtls:
        raise RuntimeError(
            "The plot requires the median (the 0.5 quantile) to be recorded."
        )
    if not len(qtls) % 2 == 1:  # If list length is not odd
        raise RuntimeError(
            "The number of quantiles must be odd so that there are pairs of quantiles around the median."
        )

    n_bands = np.floor_divide(len(qtls), 2)
    band_bounds = []
    for i in range(n_bands):
        band_bounds.append((qtls[i], qtls[-(i + 1)]))
    return band_bounds


def plot_quantile_bands(
    g: sns.FacetGrid, x: str, y_quants: list, palette: str = "mako_r", **kwargs
) -> sns.FacetGrid:
    """Overlay shaded quantile bands on a seaborn FacetGrid.

    For each pair of symmetric quantile columns (e.g. 5th/95th, 25th/75th),
    draws a ``fill_between`` band using colours drawn from ``palette``.
    Bands are drawn from outermost to innermost so darker inner bands are
    rendered on top.

    Args:
        g: Seaborn ``FacetGrid`` to draw onto.
        x: Column name for the x-axis (e.g. ``"local_time"``).
        y_quants: Sorted list of quantile column names corresponding to the
            quantile levels (must be compatible with :func:`get_band_bounds`).
        palette: Seaborn palette name for colouring bands.  Defaults to
            ``"mako_r"``.
        **kwargs: Additional keyword arguments forwarded to ``plt.fill_between``
            (e.g. ``alpha``, ``linewidth``).

    Returns:
        The modified ``FacetGrid``.
    """
    pal = sns.color_palette(palette)

    def _plot_band(x, y_lower, y_upper, **kwargs):
        plt.fill_between(x, y_lower, y_upper, **kwargs)

    for i, (low, high) in enumerate(get_band_bounds(y_quants)):
        g.map(_plot_band, x, low, high, color=pal[i], **kwargs)
    return g
