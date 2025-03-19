import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def densify_sparse_profiles(
    profs: pd.DataFrame,
    possible_times: np.ndarray,
    grp_cols: str | list[str],
    time_col: str,
    value_cols: str | list[str] = None,
    fill_value=None,
) -> pd.DataFrame:
    """Take a sparse profile and densify it across time for all groups."""

    if isinstance(grp_cols, str):
        grp_cols = [grp_cols]

    if value_cols is not None and isinstance(value_cols, str):
        value_cols = [value_cols]

    orig_idx = set(profs.index.names)
    orig_cols = set(profs.columns)
    prf = profs.reset_index()
    if orig_idx == {None}:
        new_cols = set(prf.columns)
        prf = prf.drop(columns=new_cols.difference(orig_cols))

    uniq_grp_dict = {}
    for gcol in grp_cols:
        uniq_grp_dict[gcol] = prf[gcol].unique()
    uniq_grp_dict[time_col] = possible_times

    frame = pd.MultiIndex.from_product(
        uniq_grp_dict.values(), names=uniq_grp_dict.keys()
    )
    frame = frame.to_frame(index=False)
    mrg_sort_cols = grp_cols + [time_col]
    frame = frame.merge(prf, how="left", on=grp_cols + [time_col])
    frame = frame.set_index(mrg_sort_cols)
    frame = frame.sort_index()

    if value_cols is not None and fill_value is not None:
        for vcol in value_cols:
            frame[vcol] = frame[vcol].fillna(fill_value)
    return frame
