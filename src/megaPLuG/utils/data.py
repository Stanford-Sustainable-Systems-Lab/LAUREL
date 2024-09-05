import dask.dataframe as dd
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
    col_0_max = max(left[on[0]].max(), right_merge[on[0]].max())

    right_merge["_merge"] = get_multi_col_merger(
        right_merge, src_cols=on, col_0_max=col_0_max
    )
    left["_merge"] = get_multi_col_merger(left, src_cols=on, col_0_max=col_0_max)

    # Perform merge
    right_merge = right_merge.drop(columns=on)
    right_merge = right_merge.set_index("_merge")
    left = left.merge(right_merge, on="_merge", **kwargs)
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
