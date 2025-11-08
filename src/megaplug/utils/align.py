import cvxpy as cp
import numpy as np


def calc_intersect_alignment(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the normalized alignment between two same-size arrays."""
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
    """Calculate the Wasserstein distance between two normalized vectors on the circle."""
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
