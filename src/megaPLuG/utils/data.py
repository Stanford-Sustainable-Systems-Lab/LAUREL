from typing import Self

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd


def merge_on_int_cols(
    left: dd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    **kwargs,
) -> dd.DataFrame:
    """Merges a Pandas DataFrame onto a Dask DataFrame on one or more integer columns."""
    # Pull values out of indices
    if isinstance(on, str):
        on = [on]
    orig_left_idx = None
    for col in on:
        if col == left.index.name:
            left[col] = 0
            left = left.map_partitions(
                func=_index_to_col, tgt_col=col, meta=dd.utils.make_meta(left)
            )
            orig_left_idx = col
    right_merge = right.reset_index()

    # Get combined merge column
    col_0_max = left[on[0]].max()

    right_merge["_merge"] = get_multi_col_merger(
        right_merge, src_cols=on, col_0_max=col_0_max
    )
    left["_merge"] = get_multi_col_merger(left, src_cols=on, col_0_max=col_0_max)

    # Perform merge
    right_merge = right_merge.drop(columns=on)
    right_merge = right_merge.set_index("_merge")
    left = left.merge(right_merge, on="_merge", how="left", **kwargs)
    left = left.drop(columns="_merge")

    # Reset index, if necessary
    if left.index.name != orig_left_idx:
        if isinstance(left, pd.DataFrame):
            left = left.set_index(orig_left_idx)
        elif isinstance(left, dd.DataFrame):
            left = left.set_index(
                orig_left_idx, sorted=True
            )  # Assuming no reorder by merge
    return left


def _index_to_col(df: pd.DataFrame, tgt_col: str) -> pd.DataFrame:
    df[tgt_col] = df.index.values
    return df


def get_multi_col_merger(
    df: pd.DataFrame | dd.DataFrame, src_cols: list[str], col_0_max: int
) -> pd.Series | dd.Series:
    """Build a single column to merge on which is a combination of two columns.

    We achieve this using an encoding function (e.g. id1 * set1_size + id2). An
    alternative to this would be a hash.

    The columns may not include an index.
    """
    MAX_SUPPORTED_COLS = 2
    if len(src_cols) == 1:
        return df[*src_cols]
    elif len(src_cols) > MAX_SUPPORTED_COLS:
        raise NotImplementedError

    return df[src_cols[0]] * col_0_max + df[src_cols[1]]


def filter_by_vals_in_cols(
    df: pd.DataFrame | gpd.GeoDataFrame, params: dict
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Filter dataframe so that columns contain only the values listed in values.

    The different columns are and'ed or or'ed together based on the params.
    """
    filt_ser = np.ones(len(df), dtype=np.bool)
    keep_cols = []
    for col, filt in params["filters"].items():
        cur_ser = df[col].isin(filt["value_isin"])
        if "joining_bool" not in filt or filt["joining_bool"].upper() == "AND":
            filt_ser &= cur_ser
        elif filt["joining_bool"].upper() == "OR":
            filt_ser |= cur_ser
        keep_cols.append(col)
    if isinstance(df, gpd.GeoDataFrame):
        keep_cols += [df.geometry.name]
    filtered = df.loc[filt_ser, keep_cols]
    return filtered


def get_basic_dtype_ser(ser: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(ser):
        return ser.astype(np.int64)  # or 'int64' if appropriate
    elif pd.api.types.is_float_dtype(ser):
        return ser.astype(np.float64)
    else:
        raise RuntimeError("No available non-nullable dtype!")


class IndexIntegerizer:
    """Integerize a (multi)index of a Pandas DataFrame and recover back to original.

    This will usually accompany processing by `numba`.
    """

    _uniques: np.ndarray = None
    _idx_names: list = None
    _int_col: str = None

    def __init__(self: Self, int_col: str) -> None:
        self.int_col = int_col

    def integerize(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the original (multi)index to an integer form of itself."""
        self.idx_names = df.index.names
        codes, self.uniques = df.index.factorize()
        df = df.set_index(codes)
        df.index.name = self.int_col
        return df

    def deintegerize(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the integerized column back to the original form."""
        if len(df.index.names) > 1:
            raise RuntimeError("Only a single index level can be deintegerized.")
        idx_vals = df.index.get_level_values(self.int_col)
        df = df.set_index(self.uniques[idx_vals])
        df.index.names = self.idx_names
        return df

    @property
    def idx_names(self: Self) -> str:
        return self._idx_names

    @idx_names.setter
    def idx_names(self: Self, value: str) -> None:
        self._idx_names = value

    @property
    def int_col(self: Self) -> str:
        return self._int_col

    @int_col.setter
    def int_col(self: Self, value: str) -> None:
        self._int_col = value

    @property
    def uniques(self: Self) -> pd.DataFrame:
        return self._uniques

    @uniques.setter
    def uniques(self: Self, value: pd.DataFrame) -> None:
        self._uniques = value
