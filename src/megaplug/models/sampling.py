import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import scipy as sp
from numba import jit
from numpy.typing import NDArray

from megaplug.models.summarize import IntervalBeginSpreader
from megaplug.utils.time import total_time_units

logger = logging.getLogger(__name__)


def build_entity_mask_array(
    ids: NDArray[np.int_], n_ent: int | None = None
) -> sp.sparse.coo_array:
    """Build an array whose columns are each a mask on the observations for a particular entity.

    Args:
        ids: The array (n_obs,) of entity ids for each observation
        n_ent: The number of entities which exist

    Returns: Sparse COO Array of (n_obs, n_ent) whose columns are a mask on the observations
    belonging to a particular entity.

    Example Usage:
    obs["entity_id_compact"] = pd.Categorical(sess["entity_id"]).codes
    mask = build_entity_mask_array(obs["entity_id_compact"])
    """
    n_obs = ids.shape[0]

    if n_ent is None:
        n_ent = np.unique_values(ids).size

    data = np.ones(n_obs)
    rows = np.arange(n_obs)

    mask = sp.sparse.coo_array((data, (rows, ids)), shape=(n_obs, n_ent))
    return mask


def normalize_sparse(
    arr: sp.sparse.sparray,
    axis: int = 0,
    *,
    handle_zeros: Literal["leave", "warn", "raise"] = "leave",
) -> "sp.sparse.sparray | sp.sparse.spmatrix":
    """Normalize the rows or columns of a SciPy sparse array or matrix by their sum.

    Parameters
    ----------
    arr
        Sparse array or matrix to normalize.
    axis
        ``0`` to normalize columns, ``1`` to normalize rows.
    handle_zeros
        Policy for zero-sum rows/columns:
        - ``"leave"`` keeps the row/column all zeros.
        - ``"warn"`` keeps zeros and emits a warning.
        - ``"raise"`` raises ``ZeroDivisionError``.

    Returns
    -------
    Sparse array or matrix
        The normalized input, with type preserved.
    """
    if not sp.sparse.issparse(arr):
        raise TypeError("Expected a SciPy sparse array or matrix input")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (columns) or 1 (rows)")
    if handle_zeros not in {"leave", "warn", "raise"}:
        raise ValueError("handle_zeros must be one of {'leave', 'warn', 'raise'}")

    working = arr.tocsc(copy=True) if axis == 0 else arr.tocsr(copy=True)
    sums = np.asarray(working.sum(axis=axis)).ravel()

    zero_mask = sums == 0
    if zero_mask.any():
        if handle_zeros == "raise":
            raise ZeroDivisionError(
                "Encountered zero-sum row/column during sparse normalization",
            )
        if handle_zeros == "warn":
            warnings.warn(
                "Encountered zero-sum row/column; leaving values unchanged",
                RuntimeWarning,
                stacklevel=2,
            )

    inv = np.zeros_like(sums, dtype=np.float64)
    np.divide(1.0, sums, out=inv, where=~zero_mask)

    counts = np.diff(working.indptr)
    if counts.size:
        working.data *= np.repeat(inv, counts)

    normalized = type(arr)(working)

    return normalized


@jit
def sample_sparse_multinomial_core(
    n_arr: NDArray,
    data: NDArray,
    indices: NDArray,
    indptr: NDArray,
    loc_grp_arr: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Sample a from a multinomial distribution for each row/column of a sparse array.

    Note that this function will return no samples when there are no samples available.
    This allows it to not error out.
    """
    # Pre-allocate arrays to store the sample weights for dwells for each location
    #  We know that, at most, we'll have one entry for each expected sample. However,
    #  we may have less because the multinomial may sample any given dwell more than once.
    #  At the end of the loop we will cut off these arrays to only include those elements
    #  which have been assigned.
    n_total_samples = n_arr.sum()
    n_locs = n_arr.size
    w_data = np.zeros(shape=(n_total_samples,), dtype=np.int64)
    w_indices = np.zeros_like(w_data)
    w_indptr = np.zeros(shape=(n_locs + 1,), dtype=np.int64)
    w_cursor = 0
    w_cursor_next = 0

    for hex in range(n_arr.size):
        n = n_arr[hex]
        if n > 0:  # If any samples are to be taken
            if loc_grp_arr is not None:
                i = loc_grp_arr[hex]
            else:
                i = hex

            flat_idx_first = indptr[i]
            flat_idx_last = indptr[i + 1]
            if flat_idx_last > flat_idx_first:  # If any items available to sample
                probs = data[flat_idx_first:flat_idx_last]
                inds = indices[flat_idx_first:flat_idx_last]
                w = np.random.multinomial(n=n, pvals=probs)

                out_sel = np.nonzero(w)[0]
                w_cursor_next = w_cursor + out_sel.size
                w_data[w_cursor:w_cursor_next] = w[out_sel]
                w_indices[w_cursor:w_cursor_next] = inds[out_sel]

                w_cursor = w_cursor_next

        # Increment the indptr regardless
        w_indptr[hex + 1] = w_cursor

    # Cut off the w_arrays to match the number of actually assigned values
    w_indices = w_indices[:w_cursor]
    w_data = w_data[:w_cursor]

    return (w_data, w_indices, w_indptr)


def sample_sparse_multinomial(
    n_arr: NDArray,
    p_arr: sp.sparse.sparray,
    loc_grp_arr: NDArray | None = None,
) -> sp.sparse.sparray:
    """Sample a from a multinomial distribution for each row/column of a sparse array.

    Parameters
    ----------
    n_arr : NDArray
        1D array of sample counts for each location/column
    p_arr : sp.sparse.sparray
        Sparse probability matrix in CSC format (dwells x locations)
    loc_grp_arr : NDArray | None
        Optional 1D array mapping each location to a group index

    Returns
    -------
    sp.sparse.sparray
        Sparse matrix of sampled counts with same shape as p_arr

    Raises
    ------
    ValueError
        If input dimensions are incompatible
    """
    # Dimension checks
    if not isinstance(n_arr, np.ndarray) or n_arr.ndim != 1:
        raise ValueError(
            f"n_arr must be a 1D numpy array, got shape {n_arr.shape if hasattr(n_arr, 'shape') else type(n_arr)}"
        )

    if not sp.sparse.issparse(p_arr):
        raise ValueError(f"p_arr must be a sparse array, got {type(p_arr)}")

    if p_arr.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"p_arr must be 2D, got shape {p_arr.shape}")

    # Check n_arr length matches number of columns in p_arr
    if n_arr.size != p_arr.shape[1] and loc_grp_arr is None:
        raise ValueError(
            f"n_arr length ({n_arr.size}) must match p_arr columns ({p_arr.shape[1]})"
        )

    # Convert to CSC format if not already
    if not isinstance(p_arr, sp.sparse.csc_matrix | sp.sparse.csc_array):
        p_arr = p_arr.tocsc()

    # Check loc_grp_arr if provided
    if loc_grp_arr is not None:
        if not isinstance(loc_grp_arr, np.ndarray) or loc_grp_arr.ndim != 1:
            raise ValueError(
                f"loc_grp_arr must be a 1D numpy array, got shape {loc_grp_arr.shape if hasattr(loc_grp_arr, 'shape') else type(loc_grp_arr)}"
            )

        if loc_grp_arr.size != n_arr.size:
            raise ValueError(
                f"loc_grp_arr length ({loc_grp_arr.size}) must match n_arr length ({n_arr.size})"
            )

        # Check that all group indices are valid for the sparse array structure
        max_grp_idx = loc_grp_arr.max()
        if max_grp_idx >= len(p_arr.indptr) - 1:
            raise ValueError(
                f"Maximum group index ({max_grp_idx}) exceeds sparse array structure (max valid: {len(p_arr.indptr) - 2})"
            )

        min_grp_idx = loc_grp_arr.min()
        if min_grp_idx < 0:
            raise ValueError(
                f"Group indices must be non-negative, got minimum: {min_grp_idx}"
            )

    # Check for negative sample counts
    if np.any(n_arr < 0):
        raise ValueError("All elements in n_arr must be non-negative")

    out_data, out_indices, out_indptr = sample_sparse_multinomial_core(
        n_arr=n_arr,
        data=p_arr.data,
        indices=p_arr.indices,
        indptr=p_arr.indptr,
        loc_grp_arr=loc_grp_arr,
    )
    out_shape = (p_arr.shape[0], n_arr.shape[0])
    out_sparse = sp.sparse.csc_array(
        (out_data, out_indices, out_indptr), shape=out_shape
    )
    return out_sparse


def _collate_sparse_diffs_core(
    diffs: NDArray,
    indices: NDArray[np.int_],
    indptr: NDArray[np.int_],
    times: NDArray[np.datetime64],
    final_time: np.datetime64,
) -> tuple[
    NDArray[np.int_],
    NDArray[np.datetime64],
    NDArray[np.timedelta64],
    NDArray,
]:
    """Core functionality for flattening and grouping sparse array diffs.

    Args:
        diffs: array of dimension (n_obs, n_profs) giving data to process
        indices: array of dimension (n_obs,) giving indices from sparse matrix
        indptr: array of dimension (n_regs + 1) giving column/row pointers
        times: array of dimension (n_events,) giving the time for all possible events
        final_time: datetime64 giving time to append when taking time diffs

    Returns
    """
    n_regs = indptr.shape[0] - 1
    n_obs = diffs.shape[0]

    reg_arr = np.zeros(shape=(n_obs,), dtype=np.int64)
    time_arr = np.zeros(shape=(n_obs,), dtype=times.dtype)

    test_time = np.ones(1, dtype=times.dtype)
    dur_dtype = (test_time - test_time).dtype
    dur_arr = np.zeros(shape=(n_obs,), dtype=dur_dtype)
    prof_arr = np.zeros_like(diffs)

    cursor = 0
    cursor_next = 0
    for reg in range(n_regs):
        idx_first = indptr[reg]
        idx_last = indptr[reg + 1]
        reg_diffs = diffs[idx_first:idx_last, :]

        cursor_next = cursor + reg_diffs.shape[0]
        prof_arr[cursor:cursor_next, :] = np.cumsum(reg_diffs, axis=0)
        time_arr[cursor:cursor_next] = times[indices[idx_first:idx_last]]
        dur_arr[cursor:cursor_next] = np.diff(
            time_arr[cursor:cursor_next], append=final_time
        )
        reg_arr[cursor:cursor_next] = reg

        cursor = cursor_next

    return reg_arr, time_arr, dur_arr, prof_arr


def collate_sparse_diffs(
    times: NDArray,
    final_time: np.datetime64,
    group_name: str,
    time_name: str,
    dur_name: str = "duration",
    validate_structure: bool = True,
    **sparses: sp.sparse.sparray,
) -> pd.DataFrame:
    """
    Collate sparse sets of difference values into a dataframe of cumsum-ed profiles.

    Args:
        times: Ordered event timestamps associated with the sparse diffs.
        final_time: Terminal timestamp used when computing trailing durations.
        validate_structure: Whether to require identical explicit elements across inputs.
        sparses: A dict of sparse arrays of shape (n_regions, n_events). The keys are
            the column names that will be used in the resulting DataFrame.
    """
    if sparses is None or len(sparses) == 0:
        raise ValueError("At least one sparse array must be passed through kwargs.")

    if any([not isinstance(obj, sp.sparse.sparray) for obj in sparses.values()]):
        raise ValueError("All kwargs must be sparse arrays.")

    ref_name = list(sparses.keys())[0]
    ref_shape = sparses[ref_name].shape
    if any([ref_shape != arr.shape for arr in sparses.values()]):
        raise ValueError("All sparse arrays must have the same shape.")

    # Harmonize sparse array types
    for name, arr in sparses.items():
        sparses[name] = arr.tocsr()

    ref_arr = sparses[ref_name]
    if validate_structure and len(sparses) > 1:
        ref_indptr = ref_arr.indptr
        ref_indices = ref_arr.indices
        mismatched = [
            name
            for name, arr in sparses.items()
            if not (
                np.array_equal(arr.indptr, ref_indptr)
                and np.array_equal(arr.indices, ref_indices)
            )
        ]
        if mismatched:
            mismatch_str = ", ".join(sorted(mismatched))
            raise ValueError(
                "All sparse arrays must share the same explicit elements; "
                f"found mismatched structure in: {mismatch_str}.",
            )

    sparse_names = list(sparses.keys())
    sparse_datas = [sparses[k].data[:, np.newaxis] for k in sparse_names]
    sparse_dtypes = [arr.dtype for arr in sparse_datas]
    if len(sparses) > 1:
        diffs = np.concatenate(sparse_datas, axis=1)
    else:
        diffs = sparse_datas[0]

    reg_arr, time_arr, dur_arr, prof_arr = _collate_sparse_diffs_core(
        diffs=diffs,
        indices=ref_arr.indices,
        indptr=ref_arr.indptr,
        times=times,
        final_time=final_time,
    )

    prof_dict = {
        group_name: reg_arr,
        time_name: time_arr,
        dur_name: dur_arr,
    }

    # Add profile columns
    for i, name in enumerate(sparse_names):
        target_dtype = sparse_dtypes[i]
        prof_dict.update({name: prof_arr[:, i].astype(target_dtype)})

    prof_df = pd.DataFrame(prof_dict)
    return prof_df


def sample_profiles(
    m_hex_expected: NDArray,
    m_hex_obs: NDArray,
    m_class_expected: NDArray,
    m_class_obs: NDArray,
    hex_class: NDArray,
    max_first_stage_options: int,
    Om_hex: sp.sparse.sparray,
    Om_class: sp.sparse.sparray,
    events_by_dwells: sp.sparse.sparray,
    region_by_hex: sp.sparse.sparray,
    event_times: NDArray,
    slice_freq: str,
    discrete_freq: str,
    dur_col: str,
    region_name: str,
    time_col: str,
    sample_self: bool = True,
    sample_class: bool = True,
    seed: int | None = None,
    **event_diffs: dict[str, NDArray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_self or sample_class:
        np.random.seed(seed=seed)

        # Integerize the number of visits expected by sampling from a Bernoulli (E[N] = p)
        m_hex_trunc = np.trunc(m_hex_expected)
        m_hex_rem = m_hex_expected - m_hex_trunc
        m_hex_add = np.random.binomial(n=1, p=m_hex_rem)
        m_hex_samp = m_hex_trunc + m_hex_add
        m_hex_samp = m_hex_samp.astype(int)

        if sample_class:
            m_expected_class = np.ceil(m_class_expected).astype(int)
            m_first_stage_class = np.minimum(
                m_class_obs, max_first_stage_options
            ).astype(int)
            if np.any(m_first_stage_class < m_expected_class):
                logger.warning(
                    "WARNING: Very few available dwells in some categories relative to expected number of dwells per location."
                )
            W_first_stage_cls = sample_sparse_multinomial(
                n_arr=m_first_stage_class, p_arr=Om_class
            )
            W_first_stage_cls.data = np.ones_like(
                W_first_stage_cls.data
            )  # To create a selector
            Om_class_reduced = W_first_stage_cls * Om_class

        if sample_self and not sample_class:
            dwells_by_region = sample_sparse_multinomial(n_arr=m_hex_samp, p_arr=Om_hex)
        elif not sample_self and sample_class:
            dwells_by_region = sample_sparse_multinomial(
                n_arr=m_hex_samp, p_arr=Om_class_reduced, loc_grp_arr=hex_class
            )
        else:  # Implicitly both sample_self and sample_hex
            m_self = np.minimum(m_hex_obs, m_hex_samp).astype(int)
            m_class = m_hex_samp - m_self
            W_hex = sample_sparse_multinomial(n_arr=m_self, p_arr=Om_hex)
            W_class = sample_sparse_multinomial(
                n_arr=m_class, p_arr=Om_class_reduced, loc_grp_arr=hex_class
            )
            dwells_by_region = W_hex + W_class
    else:
        logger.warning(
            "Neither self nor class selected for sampling. Returning unsampled raw dwells instead."
        )
        dwells_by_region = Om_hex.copy()
        dwells_by_region.data = np.ones_like(dwells_by_region.data)

    # Sample sparse profiles based on sampled dwells
    event_sel = (events_by_dwells @ dwells_by_region @ region_by_hex).T
    sparse_profs = {k: diff * event_sel for k, diff in event_diffs.items()}

    profs_df = collate_sparse_diffs(
        times=event_times,
        final_time=(pd.Timestamp(0) + pd.Timedelta(slice_freq)).to_numpy(),
        group_name=region_name,
        time_name=time_col,
        dur_name=dur_col,
        **sparse_profs,
    )

    prof_cols = list(event_diffs.keys())
    cums = calculate_value_time_units(
        profs=profs_df,
        group_cols=[region_name],
        dur_col=dur_col,
        prof_cols=prof_cols,
    )

    discs = discretize_sparse_profiles(
        profs=profs_df,
        time_col=time_col,
        dur_col=dur_col,
        prof_cols=prof_cols,
        group_cols=[region_name],
        freq=discrete_freq,
    )
    return discs, cums


def calculate_value_time_units(
    profs: pd.DataFrame,
    group_cols: list[str],
    dur_col: str,
    prof_cols: list[str],
    time_unit: str = "1h",
) -> pd.DataFrame:
    """Calculate the total value-[time units] (e.g. kWh) for each region-profile pair."""
    tot_hrs = total_time_units(profs[dur_col], unit=time_unit)
    cum_cols = {col: profs[col] * tot_hrs for col in prof_cols}

    for gcol in group_cols:
        cum_cols.update({gcol: profs[gcol]})

    val_hrs = pd.concat(cum_cols, axis=1)

    totals_df = val_hrs.groupby(group_cols)[prof_cols].sum()
    return totals_df


def discretize_sparse_profiles(
    profs: pd.DataFrame,
    time_col: str,
    dur_col: str,
    prof_cols: list[str],
    group_cols: list[str],
    tz_col: str | None = None,
    freq: str = "1h",
) -> pd.DataFrame:
    """Discretize profiles by region and time grouping."""
    # First drop the observations with no duration or zero power
    is_na_dur = profs[dur_col].isna()
    any_nonzero = (profs.loc[:, prof_cols] != 0).any(axis=1)
    nonzero = profs.loc[~is_na_dur & any_nonzero, :]

    if tz_col is not None:
        grp_cols = group_cols + [tz_col]
    else:
        grp_cols = group_cols
    spreader = IntervalBeginSpreader(
        time_col=time_col,
        dur_col=dur_col,
        value_cols=prof_cols,
        group_cols=grp_cols,
        freq=freq,
    )
    nonzero_exp = spreader.spread(nonzero)

    grouper = grp_cols + [pd.Grouper(key=time_col, freq=freq)]
    gpby = nonzero_exp.groupby(grouper, sort=False, observed=True)
    grped_nonzero = gpby[prof_cols].max()
    grped_nonzero = grped_nonzero.reset_index()
    return grped_nonzero
