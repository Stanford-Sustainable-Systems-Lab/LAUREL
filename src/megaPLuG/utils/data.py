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


class ColumnIntegerizer:
    """Integerize an arbitrary column of a Pandas DataFrame and recover back to original.

    This will usually accompany processing by `numba`.
    """

    _orig_col: str
    _corresp: pd.DataFrame

    def __init__(self: Self, orig_col: str) -> None:
        self.orig_col = orig_col

    def integerize(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the original column to an integer form of itself while retaining the name."""
        if self.orig_col not in df.columns.tolist():
            raise RuntimeError(
                f"{self.orig_col} not found in the columns of the dataframe."
            )

        col_order = df.columns.tolist()
        orig_idx = df.index.names
        if orig_idx != [None]:
            df = df.reset_index()
        cat_ser = df[self.orig_col].astype("category")
        self.corresp = pd.DataFrame(
            {
                self.orig_col: cat_ser.cat.categories,
                "code": np.arange(len(cat_ser.cat.categories)),
            }
        )

        df = df.merge(self.corresp, how="left", on=self.orig_col)
        df = df.drop(columns=[self.orig_col])
        df = df.rename(columns={"code": self.orig_col})
        if orig_idx != [None]:
            df = df.set_index(orig_idx)
        df = df.loc[:, col_order]
        return df

    def deintegerize(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the integerized column back to the original form."""
        col_order = df.columns.tolist()
        orig_idx = df.index.names
        if orig_idx != [None]:
            df = df.reset_index()
        df = df.rename(columns={self.orig_col: "code"})
        df = df.merge(self.corresp, how="left", on="code")
        df = df.drop(columns=["code"])
        if orig_idx != [None]:
            df = df.set_index(orig_idx)
        df = df.loc[:, col_order]
        return df

    @property
    def orig_col(self: Self) -> str:
        return self._orig_col

    @orig_col.setter
    def orig_col(self: Self, value: str) -> None:
        self._orig_col = value

    @property
    def corresp(self: Self) -> pd.DataFrame:
        return self._corresp

    @corresp.setter
    def corresp(self: Self, value: pd.DataFrame) -> None:
        self._corresp = value
