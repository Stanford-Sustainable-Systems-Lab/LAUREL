import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from megaPLuG.models.group_times import AdaptiveTimeGrouper

MEDIAN = 0.5


def get_band_bounds(qtls: list[float] | pd.Index) -> list[tuple[float, float]]:
    """Get the error band bounds for plotting profile quantiles."""
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
    pal = sns.color_palette(palette)

    def _plot_band(x, y_lower, y_upper, **kwargs):
        plt.fill_between(x, y_lower, y_upper, **kwargs)

    for i, (low, high) in enumerate(get_band_bounds(y_quants)):
        g.map(_plot_band, x, low, high, color=pal[i], **kwargs)
    return g


def densify_profiles(
    sparse: pd.DataFrame,
    time_col: str,
    grp_cols: list[object],
    value_cols: list[object],
    freq: str,
    dur: str = "1d",
    fill_value=0.0,
) -> pd.DataFrame:
    """Densify sparse observed charging profiles across typical days.

    Creates a complete time series profile by filling missing time points with
    specified fill values. For each group defined by grp_cols, generates all
    possible time combinations within the specified duration and frequency, then
    merges with the original sparse data.

    Args:
        sparse: Sparse DataFrame containing observed charging data with missing
            time points. Can have grouping columns in either the columns or index.
        time_col: Name of the column containing time/datetime values that will
            be densified.
        grp_cols: List of column names to group by. These columns define the
            groups for which complete time series will be generated. Can reference
            columns in either sparse.columns or sparse.index.names.
        value_cols: List of column names containing the values to be preserved
            and filled. These are the actual data columns that will be densified.
        freq: Frequency string for time series generation (e.g., "15min", "1H").
            Must be a valid pandas frequency string.
        dur: Duration string defining the total time span for each profile
            (default: "1d"). Must be a valid pandas timedelta string.
        fill_value: Value to use for filling missing time points (default: 0.0).
            Can be any scalar value compatible with the value_cols dtypes.

    Returns:
        Complete DataFrame with densified profiles indexed by grp_cols + [time_col].
        All missing time points within each group are filled with fill_value.
        The resulting DataFrame has a MultiIndex with grouping columns and time.

    Raises:
        RuntimeError: If any column in grp_cols is not found in sparse.columns
            or sparse.index.names.

    Example:
        >>> sparse_df = pd.DataFrame({
        ...     'group_id': [1, 1, 2, 2],
        ...     'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 00:30',
        ...                                 '2023-01-01 00:15', '2023-01-01 00:45']),
        ...     'power': [10.0, 15.0, 8.0, 12.0]
        ... })
        >>> dense_df = densify_profiles(
        ...     sparse=sparse_df,
        ...     time_col='timestamp',
        ...     grp_cols=['group_id'],
        ...     value_cols=['power'],
        ...     freq='15min',
        ...     dur='1H'
        ... )
        >>> # Returns complete 15-minute intervals for each group with missing values filled
    """
    # Build all possible time combinations for a single group
    grper = AdaptiveTimeGrouper(
        time_col=time_col,
        tz_col="tz",
        start_time=pd.Timestamp(0),
        end_time=pd.Timestamp(0) + pd.Timedelta(dur) - pd.Timedelta(freq),
        possible_tzs=["local_time"],
        freq=freq,
    )
    all_classes = grper.get_possible_obs_counts().reset_index()
    all_classes = all_classes.drop(columns=["tz", grper.count_col])
    all_classes[time_col] = all_classes[time_col].dt.tz_localize(None)

    # Apply those time combinations to all groups
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

    components.append(all_classes[time_col])
    sub_frame = pd.MultiIndex.from_product(components).to_frame(index=False)

    # Merge on the original sparse profiles and fill missing values
    idx_cols = grp_cols + [time_col]
    mrg = sparse.reset_index().loc[:, idx_cols + value_cols]
    profs_frame = sub_frame.merge(mrg, how="left", on=idx_cols)
    profs_frame = profs_frame.set_index(idx_cols)
    profs_frame = profs_frame.fillna(fill_value)
    return profs_frame
