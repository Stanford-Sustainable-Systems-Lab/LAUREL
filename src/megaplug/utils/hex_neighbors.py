"""Neighbor-embedding utility for H3 hexagonal grids.

Computes spatially smoothed feature vectors by averaging the embeddings of
neighboring H3 cells.  Used in :mod:`describe_locations` to augment per-hex
employment matrices with information from adjacent hexes before K-Means
clustering, reducing the effect of sparse or missing establishment data in
individual cells.

The averaging is implemented via sparse matrix multiplication:
``neighbor_matrix @ embeddings / n_neighbors``, where the neighbor matrix is
built once and reused.  Hexes whose neighbors fall outside the observed set
are averaged over only the neighbors that exist.

Key design decisions
--------------------
- **Sparse CSR matrix**: The neighborhood structure is stored as a
  ``scipy.sparse.csr_array`` (one row per hex, non-zero entries at neighbor
  positions) so the matrix-multiply is O(n × avg_neighbors) rather than O(n²).
- **Fixed denominator**: The denominator is taken from the neighbor count of the
  *first* hex (assumed uniform across the grid), which avoids a per-row division
  but means edge hexes with fewer in-set neighbors receive a slightly downweighted
  average.
"""

import h3.api.numpy_int as h3
import numpy as np
import scipy as sp


def get_neighbor_embeddings(
    hexes: np.ndarray[np.uint64],
    embs: np.ndarray,
    include_center: bool = False,
    distance: int = 1,
) -> np.ndarray:
    """Average the feature embeddings of the neighbors of each H3 hexagon.

    For each hex in ``hexes``, looks up its neighbors within ``distance`` rings,
    finds those neighbors that exist in ``hexes``, and averages their rows in
    ``embs``.  Hexes with no in-set neighbors receive an all-zero embedding.

    The neighborhood adjacency structure is built as a sparse CSR matrix and
    the averaging is performed as a single sparse matrix multiply.

    Args:
        hexes: 1-D uint64 array of H3 cell IDs (one per observation).
        embs: 2-D float array of shape ``(n_obs, n_features)`` — the embedding
            for each hex in the same order as ``hexes``.
        include_center: If ``True``, include the hex itself when averaging its
            neighbors.  Defaults to ``False`` (ring neighbors only).
        distance: Number of H3 grid rings to include as neighbors.
            ``distance=1`` means the 6 immediately adjacent hexes.

    Returns:
        2-D float array of shape ``(n_obs, n_features)`` — the averaged
        neighbor embeddings, one row per hex in ``hexes``.
    """
    n_obs = hexes.size
    assert (
        n_obs == embs.shape[0]
    ), "Number of observed hexes and number of embeddings do not match."
    assert np.isdtype(
        hexes.dtype, np.uint64
    ), "Hexes do not have the correct (uint64) dtype."

    # Set up fast correspondence
    hex_to_idx = {hex: np.uint(idx) for idx, hex in enumerate(hexes)}

    # Build sparse matrix values
    indptr = np.zeros(shape=n_obs + 1, dtype=np.uint64)
    cur_ptr = 0
    ind_ls = []
    for hex, it in hex_to_idx.items():
        cur_idxs = get_ngbr_idxs(
            hex,
            hex_to_idx=hex_to_idx,
            include_center=include_center,
            distance=distance,
        )
        ind_ls.extend(cur_idxs)  # add the column indices for this row

        n_idxs = len(cur_idxs)
        next_ptr = cur_ptr + n_idxs
        indptr[it + 1] = next_ptr
        cur_ptr = next_ptr

    # Form up the sparse matrix
    indices = np.array(ind_ls, dtype=np.uint64)
    data = np.ones_like(indices, dtype=np.uint8)  # data are all ones
    ngbr_arr = sp.sparse.csr_array((data, indices, indptr), shape=(n_obs, n_obs))

    # Calculate average neighbor embeddings
    denom = get_ngbrs(
        hex=hexes[0], include_center=include_center, distance=distance
    ).size
    ngbr_embs = ngbr_arr @ embs
    ngbr_embs = ngbr_embs / denom
    return ngbr_embs


def get_ngbr_idxs(
    hex: np.uint,
    hex_to_idx: dict[np.uint, np.uint],
    include_center: bool,
    distance: int,
) -> list[np.uint]:
    """Return the row indices (within ``hexes``) of the neighbors of ``hex``.

    Only neighbors that appear in ``hex_to_idx`` (i.e. have an observed
    embedding) are returned; out-of-set neighbors are silently skipped.

    Args:
        hex: The H3 uint64 cell ID to look up neighbors for.
        hex_to_idx: Mapping from H3 cell ID to its position in the ``hexes``
            array.
        include_center: Whether to include ``hex`` itself.
        distance: Ring distance passed to :func:`get_ngbrs`.

    Returns:
        List of integer indices into the ``hexes``/``embs`` arrays.
    """
    ngbrs = get_ngbrs(hex=hex, include_center=include_center, distance=distance)
    idx_exist = []
    for nb in ngbrs:
        cur_idx = hex_to_idx.get(nb, None)
        if cur_idx is not None:
            idx_exist.append(cur_idx)
    return idx_exist


def get_ngbrs(
    hex: np.uint, include_center: bool, distance: int
) -> np.ndarray[np.uint64]:
    """Return the H3 neighbor cell IDs for a single hex at a given ring distance.

    Args:
        hex: H3 uint64 cell ID.
        include_center: If ``True``, include ``hex`` in the returned set.
        distance: Number of grid rings.  ``distance=1`` returns the 6 adjacent
            hexes (or 7 with ``include_center=True``).

    Returns:
        1-D uint64 array of neighbor cell IDs.
    """
    if include_center:
        ngbrs = h3.grid_disk(hex, distance)
    elif distance == 1:
        ngbrs = h3.grid_ring(hex, distance)
    else:
        ngbrs = h3.grid_disk(hex, distance)
        ngbrs = np.setdiff1d(ngbrs, hex)
    return ngbrs
