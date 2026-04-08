"""Bitmask encoding/decoding for charging-mode availability vectors.

The charging-choice algorithm (see :mod:`laurel.models.charging_algorithms`)
needs to know, for each dwell, which of the ``N`` charging modes are available
(depot, destination, truck stop, etc.).  Storing one boolean column per mode
would be expensive at scale; instead, the availability of all modes is packed
into a single ``uint64`` bitmask column, where bit ``j`` is set if mode ``j``
is available.

This module provides three functions for working with these bitmasks:

- :func:`bool_arr_to_bits` — vectorised encode: boolean array → uint64 bitmask(s).
- :func:`bits_to_bool_arr` — vectorised decode: uint64 bitmask(s) → 2-D boolean array.
- :func:`bits_to_bool_vec` — scalar Numba JIT decode: single uint64 → 1-D boolean vector.

The ``MAX_CHARGE_MODES = 64`` limit follows from the uint64 representation.
"""

import numpy as np
from numba import jit

MAX_CHARGE_MODES = 64  # Bitmask supports up to 64 modes (uint64)


def bool_arr_to_bits(arr: np.ndarray) -> np.ndarray:
    """Vectorized encode of boolean array (0D/1D/2D) to per-row uint64 bitmasks.

    Normalizes dimensionality to 2D, then uses NumPy broadcasting with powers-of-two
    weights and a row-wise sum to compute the bitmask. Returns a 1D uint64 mask array.
    Empty inputs handled gracefully.
    """
    # Normalize dimensionality
    if arr.ndim == 0:
        arr2 = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr2 = arr.reshape(1, arr.shape[0])
    elif arr.ndim == 2:  # noqa: PLR2004
        arr2 = arr
    else:  # Unsupported higher dimensional input
        raise ValueError("bool_arr_to_bits expects 0-D, 1-D, or 2-D input")

    n_items = arr2.shape[0]
    n_opts = arr2.shape[1]
    # Early exits
    if n_items == 0:
        return np.zeros(0, dtype=np.uint64)
    if n_opts == 0:
        return np.zeros(n_items, dtype=np.uint64)
    if n_opts > MAX_CHARGE_MODES:
        raise ValueError("Bitmask encoding supports at most 64 options")

    # Ensure boolean dtype without copying if already bool
    arr_bool = arr2.astype(bool, copy=False)
    # Powers of two for each mode position (LSB = mode index 0)
    powers = np.left_shift(np.uint64(1), np.arange(n_opts, dtype=np.uint64))
    # Broadcast multiply and reduce across modes
    out = (arr_bool.astype(np.uint64, copy=False) * powers).sum(axis=1, dtype=np.uint64)
    # Ensure correct dtype for empty/edge cases
    if out.dtype != np.uint64:
        out = out.astype(np.uint64, copy=False)
    return out


def bits_to_bool_arr(bits: np.ndarray, n_modes: int) -> np.ndarray:
    """Vectorized decode of uint64 bitmask(s) into a 2D boolean availability array.

    Parameters
    ----------
    bits : np.ndarray
        0-D or 1-D array of uint64 bitmasks. If 0-D/1-D, treated as a single row or
        vector of rows respectively. A 2-D array is not semantically meaningful here
        (would imply matrix of masks) and will raise.
    n_modes : int
        Number of mode positions (columns) to decode. Must be <= MAX_CHARGE_MODES.

    Returns
    -------
    np.ndarray
        2-D boolean array of shape (n_items, n_modes) where element [i, j] is True if
        bit j of bits[i] is set.
    """
    if n_modes < 0:
        raise ValueError("n_modes must be non-negative")
    if n_modes > MAX_CHARGE_MODES:
        raise ValueError("Bitmask decoding supports at most 64 modes")

    # Normalize dimensionality of bits input
    if bits.ndim == 0:
        bits2 = bits.reshape(1)
    elif bits.ndim == 1:
        bits2 = bits
    else:
        raise ValueError("bits_to_bool_arr expects 0-D or 1-D bitmask input")

    n_items = bits2.shape[0]
    # Early exits
    if n_items == 0:
        return np.zeros((0, n_modes), dtype=np.bool_)
    if n_modes == 0:
        return np.zeros((n_items, 0), dtype=np.bool_)

    # Ensure uint64 dtype for bitwise ops
    masks = bits2.astype(np.uint64, copy=False)
    # Powers-of-two bit positions (LSB corresponds to mode index 0)
    powers = np.left_shift(np.uint64(1), np.arange(n_modes, dtype=np.uint64))
    # Broadcast bit-test across all rows and mode positions
    out = (masks[:, None] & powers[None, :]) != 0
    return out.astype(np.bool_, copy=False)


@jit
def bits_to_bool_vec(bitmask: np.uint64, n_modes: int) -> np.ndarray:
    """Decode a single uint64 bitmask to a 1-D boolean vector (Numba nopython).

    Parameters
    ----------
    bitmask : np.uint64 or int
        Single bitmask whose bits indicate availability of modes.
    n_modes : int
        Number of mode positions (length of the output). Must be <= MAX_CHARGE_MODES.

    Returns
    -------
    np.ndarray
        1-D boolean array of length n_modes where element j is True if bit j is set.
    """
    if n_modes < 0:
        raise ValueError("n_modes must be non-negative")
    if n_modes > MAX_CHARGE_MODES:
        raise ValueError("Bitmask decoding supports at most 64 modes")
    if n_modes == 0:
        return np.zeros(0, dtype=np.bool_)

    mask = np.uint64(bitmask)
    out = np.zeros(n_modes, dtype=np.bool_)
    for j in range(n_modes):
        out[j] = (mask & (np.uint64(1) << np.uint64(j))) != 0
    return out
