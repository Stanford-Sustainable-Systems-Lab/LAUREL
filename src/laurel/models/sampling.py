"""Bootstrap load-profile assembly for the ``evaluate_impacts`` pipeline (Model Module 6).

Assembles per-substation peak-load and energy estimates by bootstrap sampling
across the telematics-observed dwell population for each hexagonal TAZ.  The
central function is :func:`sample_profiles`, which performs a two-stage
inverse-propensity-weighted draw:

- **Stage 1 (class-level draw)**: For each hex cell, draw candidate dwells
  from the observed dwell population in the same freight-activity class, using
  normalised inverse-propensity weights ``Om_class``.  This ensures that each
  hex's sample is drawn from a class-representative pool even when the hex
  itself has few direct observations.
- **Stage 2 (self draw)**: Supplement with dwells drawn directly from the
  hex's own observed dwell population, up to the available count.

Supporting sparse-matrix utilities
-----------------------------------
- :func:`build_entity_mask_array`: builds a ``(n_obs, n_ent)`` CSC indicator
  array mapping observations to entities.
- :func:`normalize_sparse`: column- or row-normalises a sparse array, with
  configurable zero-sum handling.
- :func:`sample_sparse_multinomial` / :func:`sample_sparse_multinomial_core`:
  JIT-compiled multinomial draw from each column of a sparse probability
  matrix.
- :func:`collate_sparse_diffs` / :func:`_collate_sparse_diffs_core`: converts
  sparse power-difference arrays to cumulative load-profile DataFrames (one
  region × event row per entry).
- :func:`calculate_value_time_units`, :func:`calculate_peak_units`,
  :func:`discretize_sparse_profiles`: summarise profiles into kWh totals, peak
  kW, and discretised time-series DataFrames.

Key design decisions
--------------------
- **Sparse representation**: each possible charging event is stored as a row
  in the event-observation matrix; the sparse format avoids materialising
  dense ``(n_events × n_regions)`` arrays that would be prohibitively large
  for 52,000 substations × 100 bootstrap draws.
- **Bernoulli fractional-sample rounding**: expected dwell counts are
  non-integer; fractional parts are rounded stochastically via a Bernoulli
  draw (``Binomial(n=1, p=fractional_part)``) to avoid systematic bias.
- **JIT inner loop**: :func:`sample_sparse_multinomial_core` iterates over
  regions in a tight loop that would be slow in interpreted Python; the
  ``@jit`` decorator eliminates the per-region Python overhead.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.
"""

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import scipy as sp
from numba import jit
from numpy.typing import NDArray

from laurel.models.summarize import IntervalBeginSpreader
from laurel.utils.time import total_time_units

logger = logging.getLogger(__name__)


def build_entity_mask_array(
    ids: NDArray[np.int_], n_ent: int | None = None
) -> sp.sparse.coo_array:
    """Build a sparse binary indicator matrix mapping observations to entities.

    Constructs a ``(n_obs, n_ent)`` COO sparse array where entry ``[i, j]``
    is 1 if observation ``i`` belongs to entity ``j`` and 0 otherwise.
    Used to construct inverse-propensity weight matrices in
    :func:`sample_profiles`.

    Args:
        ids: 1-D integer array of length ``n_obs`` giving the entity index for
            each observation.  Entity indices must be in ``[0, n_ent)``.
        n_ent: Total number of entities (number of columns in the output).  If
            ``None``, inferred as ``len(np.unique(ids))``.

    Returns:
        Sparse COO array of shape ``(n_obs, n_ent)`` with ``1.0`` entries.

    Example::

        obs["entity_id_compact"] = pd.Categorical(obs["entity_id"]).codes
        mask = build_entity_mask_array(obs["entity_id_compact"].values)
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
    """JIT-compiled inner loop: draw multinomial samples from each column of a CSC array.

    Iterates over locations (columns of the probability matrix), draws
    ``n_arr[hex]`` samples from the probability distribution stored in that
    column, and writes the non-zero counts into output CSC arrays.  Pre-allocates
    the output arrays to the maximum possible size (``n_arr.sum()``) and trims
    at the end.

    When ``loc_grp_arr`` is not ``None``, each location ``hex`` samples from
    the column indexed by ``loc_grp_arr[hex]`` in the probability matrix
    (class-level pooling).  Locations with zero samples requested are skipped,
    and locations for which the probability column is empty (no eligible
    observations) produce no output entries.

    Args:
        n_arr: 1-D integer array of sample counts for each location.
        data: CSC ``data`` array of the probability matrix (non-zero values).
        indices: CSC ``indices`` array (row indices of non-zeros).
        indptr: CSC ``indptr`` array (column pointers).
        loc_grp_arr: Optional 1-D integer array mapping each location to its
            class-pool column index.  If ``None``, each location uses its own
            column.

    Returns:
        Three-tuple ``(w_data, w_indices, w_indptr)`` — the CSC storage arrays
        for the output sparse sample-count matrix of shape
        ``(n_obs, len(n_arr))``.
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
    """Draw multinomial samples from each column of a sparse probability matrix.

    Validates inputs, converts to CSC format if needed, and delegates to the
    JIT-compiled :func:`sample_sparse_multinomial_core`.

    Args:
        n_arr: 1-D integer array of length ``n_locs`` giving the number of
            samples to draw for each location.
        p_arr: Sparse probability matrix of shape ``(n_obs, n_locs)`` (or
            ``(n_obs, n_classes)`` when ``loc_grp_arr`` is provided).  Columns
            must sum to 1 (call :func:`normalize_sparse` beforehand).
        loc_grp_arr: Optional 1-D integer array of length ``n_locs`` mapping
            each location to a column index in ``p_arr``.  Used for
            class-level pooling where many locations share one probability
            distribution.

    Returns:
        Sparse CSC array of shape ``(n_obs, n_locs)`` with integer sample
        counts.

    Raises:
        ValueError: If ``n_arr`` is not 1-D, ``p_arr`` is not 2-D sparse, or
            their dimensions are incompatible; if ``loc_grp_arr`` has invalid
            indices or length; if ``n_arr`` contains negative values.
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
    """Inner loop: convert sparse region-event power differences to cumulative profiles.

    For each region (column in the CSR layout given by ``indptr``), extracts
    the subset of events belonging to that region, takes a cumulative sum of
    the power differences along the profile axis to recover instantaneous
    power, and computes the duration of each event as the time delta to the
    next event (with ``final_time`` appended as a sentinel).

    Args:
        diffs: ``(n_obs, n_profs)`` array of power-difference values at each
            sparse event entry.  ``n_profs`` is the number of profile columns
            (e.g. one per bootstrap draw or quantile).
        indices: ``(n_obs,)`` CSR row-index array giving the event-time index
            for each non-zero entry.
        indptr: ``(n_regs + 1,)`` CSR column-pointer array.
        times: ``(n_events,)`` datetime64 array of all possible event
            timestamps (indexed by ``indices``).
        final_time: Sentinel datetime64 appended when computing the trailing
            duration for the last event in each region.

    Returns:
        Four-tuple ``(reg_arr, time_arr, dur_arr, prof_arr)`` where:

        - ``reg_arr`` (``n_obs,``): region index for each output row.
        - ``time_arr`` (``n_obs,``): event timestamp for each row.
        - ``dur_arr`` (``n_obs,``): timedelta to the next event (or
          ``final_time``).
        - ``prof_arr`` (``n_obs, n_profs``): cumulative power at each event.
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
    """Convert sparse power-difference arrays to a long-format cumulative-profile DataFrame.

    Validates that all input sparse arrays share the same shape and (if
    ``validate_structure=True``) the same explicit non-zero structure, stacks
    their ``data`` vectors column-wise, converts to CSR, and calls
    :func:`_collate_sparse_diffs_core` to recover instantaneous power profiles.

    Args:
        times: 1-D array of event timestamps; the ``i``-th element is the
            timestamp associated with row ``i`` of the sparse arrays.
        final_time: Sentinel timestamp appended when computing trailing event
            durations.
        group_name: Column name for the region index in the output DataFrame.
        time_name: Column name for the event timestamps in the output DataFrame.
        dur_name: Column name for the inter-event durations.  Defaults to
            ``"duration"``.
        validate_structure: If ``True``, raises a ``ValueError`` when any two
            input sparse arrays have different CSR sparsity patterns.
        **sparses: Named sparse arrays, each of shape ``(n_regions, n_events)``.
            Keys become the power-profile column names in the output DataFrame.

    Returns:
        Long-format DataFrame with columns ``[group_name, time_name, dur_name,
        *profile_columns]``, one row per (region, event) pair.

    Raises:
        ValueError: If no arrays are provided, any value is not a sparse
            array, shapes differ, or structures differ when validation is
            enabled.
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
    summary_suffixes: dict[str, str],
    region_name: str,
    time_col: str,
    sample_self: bool = True,
    sample_class: bool = True,
    seed: int | None = None,
    **event_diffs: dict[str, NDArray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap-sample per-region load profiles using inverse-propensity weighting.

    Implements the two-stage sampling strategy for Model Module 6:

    1. **(Optional) Class-level first stage**: for each hex, pre-select up to
       ``max_first_stage_options`` candidate dwells from the class-level pool
       by drawing from ``Om_class`` and constructing a restricted probability
       matrix ``Om_class_reduced``.
    2. **Self draw**: draw ``min(m_hex_obs, m_hex_samp)`` dwells from the hex's
       own pool ``Om_hex``.
    3. **Class draw**: draw the remaining ``m_hex_samp - m_self`` dwells from
       the restricted class-level pool.
    4. Combine the two draws, multiply through the ``events_by_dwells @
       dwells_by_region @ region_by_hex`` chain to obtain sampled event
       indicator matrices, and call :func:`collate_sparse_diffs` to produce
       load profiles.
    5. Compute kWh totals (:func:`calculate_value_time_units`), peak kW
       (:func:`calculate_peak_units`), and hourly discretised profiles
       (:func:`discretize_sparse_profiles`).

    Args:
        m_hex_expected: ``(n_hex,)`` float array of expected dwell counts per
            hex under this scenario.
        m_hex_obs: ``(n_hex,)`` integer array of observed dwell counts per hex.
        m_class_expected: ``(n_classes,)`` float array of expected dwell counts
            per freight-activity class.
        m_class_obs: ``(n_classes,)`` integer array of observed dwell counts per
            class.
        hex_class: ``(n_hex,)`` integer array mapping each hex to its class
            index.
        max_first_stage_options: Maximum number of candidates to pre-select
            from each class pool in the first stage.
        Om_hex: ``(n_dwells, n_hex)`` sparse probability matrix (hex-level
            inverse-propensity weights, column-normalised).
        Om_class: ``(n_dwells, n_classes)`` sparse probability matrix
            (class-level weights, column-normalised).
        events_by_dwells: ``(n_events, n_dwells)`` sparse indicator matrix
            mapping events to the dwell that generated them.
        region_by_hex: ``(n_regions, n_hex)`` sparse indicator matrix mapping
            hex cells to aggregation regions (e.g. substations).
        event_times: ``(n_events,)`` datetime64 array of event timestamps.
        slice_freq: Pandas frequency string used as the time-slice period for
            profile wrapping (passed as ``final_time`` in
            :func:`collate_sparse_diffs`).
        discrete_freq: Pandas frequency string for the discretised output
            profiles (passed to :func:`discretize_sparse_profiles`).
        dur_col: Column name for event durations in intermediate DataFrames.
        summary_suffixes: Dict with keys ``"cumul"`` and ``"peak"`` giving the
            suffixes for energy and peak columns in the summary DataFrame.
        region_name: Column name for the region identifier in outputs.
        time_col: Column name for event timestamps in outputs.
        sample_self: If ``True``, include the self-draw stage.
        sample_class: If ``True``, include the class-draw stage.
        seed: Random seed for reproducibility.  Defaults to ``None``.
        **event_diffs: Named sparse ``(n_obs, n_events)`` arrays of power
            differences.  Keys become the profile column names in the output.

    Returns:
        Two-tuple ``(discs, summs)`` where:

        - ``discs``: discretised load-profile DataFrame from
          :func:`discretize_sparse_profiles`.
        - ``summs``: summary DataFrame with kWh and peak-kW columns per
          region, joined from :func:`calculate_value_time_units` and
          :func:`calculate_peak_units`.
    """
    if sample_self or sample_class:
        np.random.seed(seed=seed)  # Note: This seed only applies outside of Numba

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

    peaks = calculate_peak_units(
        profs=profs_df,
        group_cols=[region_name],
        prof_cols=prof_cols,
    )

    summs = cums.join(
        peaks,
        how="outer",
        lsuffix=summary_suffixes["cumul"],
        rsuffix=summary_suffixes["peak"],
    )

    discs = discretize_sparse_profiles(
        profs=profs_df,
        time_col=time_col,
        dur_col=dur_col,
        prof_cols=prof_cols,
        group_cols=[region_name],
        freq=discrete_freq,
    )
    return discs, summs


def calculate_value_time_units(
    profs: pd.DataFrame,
    group_cols: list[str],
    dur_col: str,
    prof_cols: list[str],
    time_unit: str = "1h",
) -> pd.DataFrame:
    """Calculate total energy (kWh) per region by integrating power over time.

    Computes ``value × duration_in_units`` for each row, then sums by group.

    Args:
        profs: Long-format profile DataFrame with one event per row.
        group_cols: Columns to group by (typically ``[region_name]``).
        dur_col: Column containing event durations (``pd.Timedelta``).
        prof_cols: Power-value columns to integrate.
        time_unit: Pandas frequency string for the time unit (denominator of
            the duration conversion).  Defaults to ``"1h"`` for kWh.

    Returns:
        DataFrame indexed by ``group_cols`` with one column per entry in
        ``prof_cols``, containing the summed energy values.
    """
    tot_hrs = total_time_units(profs[dur_col], unit=time_unit)
    cum_cols = {col: profs[col] * tot_hrs for col in prof_cols}

    for gcol in group_cols:
        cum_cols.update({gcol: profs[gcol]})

    val_hrs = pd.concat(cum_cols, axis=1)

    totals_df = val_hrs.groupby(group_cols)[prof_cols].sum()
    return totals_df


def calculate_peak_units(
    profs: pd.DataFrame,
    group_cols: list[str],
    prof_cols: list[str],
) -> pd.DataFrame:
    """Return the peak power (kW) per region by taking the group-wise maximum.

    Args:
        profs: Long-format profile DataFrame.
        group_cols: Columns to group by (typically ``[region_name]``).
        prof_cols: Power-value columns to aggregate.

    Returns:
        DataFrame indexed by ``group_cols`` with one column per entry in
        ``prof_cols``, containing the maximum observed value.
    """
    peaks_df = profs.groupby(group_cols)[prof_cols].max()
    return peaks_df


def discretize_sparse_profiles(
    profs: pd.DataFrame,
    time_col: str,
    dur_col: str,
    prof_cols: list[str],
    group_cols: list[str],
    tz_col: str | None = None,
    freq: str = "1h",
) -> pd.DataFrame:
    """Aggregate load profiles onto a regular time grid by region.

    Drops zero-power or duration-less rows, spreads each remaining event onto
    all ``freq``-aligned time-bin beginnings it covers via
    :class:`~laurel.models.summarize.IntervalBeginSpreader`, then groups by
    ``group_cols + [pd.Grouper(key=time_col, freq=freq)]`` and takes the
    maximum power in each bin (appropriate for step-function profiles).

    Args:
        profs: Long-format profile DataFrame from :func:`collate_sparse_diffs`.
        time_col: Timestamp column name.
        dur_col: Duration column name.
        prof_cols: Power-value columns to aggregate.
        group_cols: Spatial grouping columns (e.g. ``[region_name]``).
        tz_col: Optional timezone column; if provided, it is included in the
            grouping so that outputs are localised.
        freq: Pandas frequency string for the output time grid.  Defaults to
            ``"1h"``.

    Returns:
        DataFrame with columns ``group_cols + [time_col] + prof_cols`` at the
        requested temporal resolution, one row per (region, time-bin) with
        at least one non-zero event.
    """
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
