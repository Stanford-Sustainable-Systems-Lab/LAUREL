"""Profile alignment metrics for comparing temporal charging load distributions.

This module provides two complementary distance/similarity measures for
comparing normalised load profiles (or any non-negative 1-D distributions):

- :func:`calc_intersect_alignment` — intersection over union of two normalised
  histograms; ranges from 0 (no overlap) to 1 (identical profiles).
- :func:`calc_wasser_circle_distance` — Earth-Mover's (Wasserstein-1) distance
  on a circular support, solved as a linear programme via ``cvxpy``.  Suitable
  for comparing time-of-day distributions where 23:00 and 01:00 are adjacent.

These metrics are used in reporting notebooks to quantify how well estimated
load profiles match observed (validation) profiles.
"""

import cvxpy as cp
import numpy as np


def calc_intersect_alignment(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the histogram-intersection similarity between two non-negative arrays.

    Normalises both arrays to sum to 1, then returns the sum of element-wise
    minima — equivalent to the fraction of the distribution mass that the two
    profiles share.  Returns ``nan`` if either array sums to zero.

    Args:
        a: Non-negative 1-D array (e.g. hourly load profile).
        b: Non-negative 1-D array of the same length as ``a``.

    Returns:
        Similarity score in [0, 1], or ``nan`` if either input is all-zero.
    """
    a_sum = np.sum(a)
    b_sum = np.sum(b)
    if a_sum == 0 or b_sum == 0:
        return np.nan
    else:
        a_norm = a / a_sum
        b_norm = b / b_sum
        align = np.sum(np.minimum(a_norm, b_norm))
        return align


def calc_wasser_circle_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Wasserstein-1 distance between two distributions on a circular domain.

    Uses a linear-programming formulation (via ``cvxpy``) with a ground metric
    that is the shorter arc-length on the unit circle, so the distance between
    bins wraps at the boundary (e.g. hour 23 and hour 1 are 2 bins apart).
    Both arrays are normalised to unit mass before solving.  Returns ``nan`` if
    either array sums to zero.

    Args:
        a: Non-negative 1-D array representing the first distribution over
            equally-spaced bins on a circle.
        b: Non-negative 1-D array of the same shape as ``a``.

    Returns:
        Wasserstein-1 distance (in units of bins), or ``nan`` if either input
        is all-zero.

    Raises:
        RuntimeError: If ``a`` and ``b`` have different shapes.
    """
    if np.squeeze(a).shape != np.squeeze(b).shape:
        raise RuntimeError("a and b must have the same shape.")

    a_sum = np.sum(a)
    b_sum = np.sum(b)
    if a_sum == 0 or b_sum == 0:
        return np.nan
    else:
        a_norm = a / a_sum
        b_norm = b / b_sum

        # Build circular distance metric
        full_circle = np.size(a)
        ratio = 2 * np.pi / full_circle
        rads = ratio * np.arange(full_circle)
        rads = rads[:, np.newaxis]

        diff = np.abs(rads - rads.T)
        cost = np.minimum(diff, 2 * np.pi - diff)
        cost = cost / ratio

        # Calculate Wasserstein distance
        T = cp.Variable(shape=cost.shape, nonneg=True)
        cons = [
            cp.sum(T, axis=0) == a_norm,
            cp.sum(T, axis=1) == b_norm,
        ]
        obj = cp.Minimize(cp.sum(cp.multiply(cost, T)))
        prob = cp.Problem(objective=obj, constraints=cons)
        distance = prob.solve()
        return distance
