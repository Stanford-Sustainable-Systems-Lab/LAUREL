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
