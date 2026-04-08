import numpy as np
import pytest

from laurel.utils.mode_masks import (
    MAX_CHARGE_MODES,
    bits_to_bool_arr,
    bits_to_bool_vec,
    bool_arr_to_bits,
)


def test_round_trip_1d_and_2d():
    # 1-D case: [True, False, True] -> 0b101
    arr1 = np.array([True, False, True])
    bits1 = bool_arr_to_bits(arr1)
    assert bits1.shape == (1,)
    assert bits1.dtype == np.uint64
    assert int(bits1[0]) == 0b101
    back1 = bits_to_bool_arr(bits1, n_modes=3)
    np.testing.assert_array_equal(back1, arr1.reshape(1, -1))

    # 2-D case: multiple rows
    arr2 = np.array(
        [[True, True, False, False], [False, True, False, True]], dtype=bool
    )
    bits2 = bool_arr_to_bits(arr2)
    assert bits2.shape == (2,)
    back2 = bits_to_bool_arr(bits2, n_modes=4)
    np.testing.assert_array_equal(back2, arr2)


def test_round_trip_0d_and_empty():
    # 0-D: single boolean
    arr0 = np.array(True)
    bits0 = bool_arr_to_bits(arr0)
    assert bits0.shape == (1,)
    back0 = bits_to_bool_arr(bits0, n_modes=1)
    np.testing.assert_array_equal(back0, np.array([[True]]))

    # Empty rows: shape (0, n)
    arr_empty = np.zeros((0, 5), dtype=bool)
    bits_empty = bool_arr_to_bits(arr_empty)
    assert bits_empty.shape == (0,)
    back_empty = bits_to_bool_arr(bits_empty, n_modes=5)
    assert back_empty.shape == (0, 5)

    # Zero modes: shape (n, 0)
    arr_zero_modes = np.zeros((3, 0), dtype=bool)
    bits_zero_modes = bool_arr_to_bits(arr_zero_modes)
    assert bits_zero_modes.shape == (3,)
    np.testing.assert_array_equal(bits_zero_modes, np.zeros(3, dtype=np.uint64))
    back_zero_modes = bits_to_bool_arr(bits_zero_modes, n_modes=0)
    assert back_zero_modes.shape == (3, 0)


def test_bits_to_bool_vec_matches_first_row():
    arr = np.array([[True, False, True, True]], dtype=bool)
    bits = bool_arr_to_bits(arr)
    vec = bits_to_bool_vec(bits[0], n_modes=arr.shape[1])
    np.testing.assert_array_equal(vec, arr[0])


def test_invalid_dimensions_raise():
    # bool_arr_to_bits should reject 3D input
    with pytest.raises(ValueError):
        _ = bool_arr_to_bits(np.zeros((1, 2, 3), dtype=bool))

    # bits_to_bool_arr should reject 2D bits input
    with pytest.raises(ValueError):
        _ = bits_to_bool_arr(np.zeros((2, 2), dtype=np.uint64), n_modes=2)


def test_mode_limit_enforced():
    # Encoding with >64 modes should raise
    with pytest.raises(ValueError):
        _ = bool_arr_to_bits(np.zeros((1, MAX_CHARGE_MODES + 1), dtype=bool))

    # Decoding with >64 modes should raise
    with pytest.raises(ValueError):
        _ = bits_to_bool_arr(np.zeros(1, dtype=np.uint64), n_modes=MAX_CHARGE_MODES + 1)
