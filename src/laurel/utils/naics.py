"""NAICS code matching utilities (Numba JIT-compiled).

Provides :func:`get_naics_leaf_class`, which maps arbitrary-depth NAICS codes
to a predefined set of "leaf" codes by iteratively truncating the code from the
right until a match is found.  Used in :mod:`describe_locations` to collapse
the full 6-to-8-digit Data Axle NAICS codes to the coarser leaf classes needed
for freight-activity clustering.
"""

import numpy as np
from numba import jit


@jit
def get_naics_leaf_class(
    codes: np.ndarray[int],
    leaves: np.ndarray[int],
    src_digits: int = 8,
    fill_leaf: int | None = None,
) -> np.ndarray[int]:
    """Map each NAICS code to the most specific matching leaf code.

    Iteratively right-truncates each code (by integer division by 10) and
    checks it against the leaf set until every code matches a leaf or has been
    truncated to zero.  This implements a hierarchical NAICS rollup: a code
    that matches no leaf at 8 digits is tried at 6, then 4, then 2 digits.

    Args:
        codes: 1-D integer array of NAICS codes to classify.
        leaves: 1-D integer array of accepted leaf codes.  Each leaf is matched
            at most once per pass, so the order of ``leaves`` does not matter.
        src_digits: Not currently used; reserved for future digit-normalisation.
        fill_leaf: If not ``None``, codes that match no leaf are assigned this
            value instead of raising.  Useful for catch-all categories.

    Returns:
        1-D integer array, same shape as ``codes``, containing the matched
        leaf code for each input.

    Raises:
        ValueError: If any code remains unmatched after full truncation and
            ``fill_leaf`` is ``None``.
    """
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
