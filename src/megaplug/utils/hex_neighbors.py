import h3.api.numpy_int as h3
import numpy as np
import scipy as sp


def get_neighbor_embeddings(
    hexes: np.ndarray[np.uint64],
    embs: np.ndarray,
    include_center: bool = False,
    distance: int = 1,
) -> np.ndarray:
    """Get average embeddings of neighboring hexes."""
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
    """For a given hex id, get all the indices of neighbor hexagons."""
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
    if include_center:
        ngbrs = h3.grid_disk(hex, distance)
    elif distance == 1:
        ngbrs = h3.grid_ring(hex, distance)
    else:
        ngbrs = h3.grid_disk(hex, distance)
        ngbrs = np.setdiff1d(ngbrs, hex)
    return ngbrs
