import numpy as np
from numba import jit


@jit
def get_naics_leaf_class(
    codes: np.ndarray[int],
    leaves: np.ndarray[int],
    src_digits: int = 8,
    fill_leaf: int | None = None,
) -> np.ndarray[int]:
    """Get the NAICS 'leaf' for each NAICS code."""
    out = np.zeros_like(codes)
    used_leaves = np.zeros_like(leaves)
    while np.any(used_leaves == 0) and np.any(codes > 0):
        for i in range(len(leaves)):  # Check all leaves
            leaf = leaves[i]
            if not used_leaves[i] == 1:
                matches = (codes == leaf) & (out == 0)
                if np.any(matches):  # If there is a match
                    out = np.where(matches, leaf, out)  # Replace out val
                    used_leaves[i] = 1
        codes = np.floor_divide(codes, 10)
    # n_digits = np.where(out == 0, 1, np.floor(np.log10(out)) + 1)
    # out = out * np.power(10, src_digits - n_digits)
    if fill_leaf is not None:
        out = np.where(out != 0, out, fill_leaf)
    elif np.any(out == 0):
        raise ValueError("At least some NAICS codes are assigned to no leaf code.")
    return out
