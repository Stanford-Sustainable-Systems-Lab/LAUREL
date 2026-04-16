"""General-purpose DataFrame utilities shared across LAUREL pipelines.

This module provides helpers for merging, filtering, type-casting, and
column-selecting pandas and Dask DataFrames, as well as a utility class for
round-tripping a DataFrame through an integer index (needed before Numba JIT
kernels) and a record-array conversion adapted from the pandas internals.

Key design decisions
--------------------
- **Index preservation**: :func:`merge_dataframes_node` resets and restores the
  original index so that callers never have to worry about index loss after a
  left-join enrichment step.
- **Dask compatibility**: every function that accepts a pandas ``DataFrame`` also
  accepts a Dask ``DataFrame`` where performance constraints require it; the
  branching is handled internally so call-sites remain uniform.
- **Integer merge key**: :func:`merge_on_int_cols` uses a Cantor-style encoding
  (``id0 * max0 + id1``) rather than a multi-column merge to avoid the expensive
  ``set_index`` that Dask requires for keyed joins on non-index columns.
"""

from __future__ import annotations

import itertools
import logging
from typing import Self

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.core import common as com
from pandas.core.dtypes.common import is_dict_like

from laurel.models.dwell_sets import DwellSet

logger = logging.getLogger(__name__)


def get_merge_params(
    merge_params: dict,
    right_df: pd.DataFrame,
    *args: list[list[str]],
) -> dict:
    """Restrict ``merge_params["keep_right_columns"]`` to columns present in ``right_df``.

    Before a merge, the config may list column names that do not exist in the
    current scenario's right DataFrame (e.g. because a column appears only in
    some scenarios).  This function intersects the desired columns with those
    actually available — including the index levels — so that
    :func:`merge_dataframes_node` never tries to select a non-existent column.
    Additional column lists passed as positional ``*args`` (e.g. the active
    ``group_columns`` param sets) are also unioned into the target set.

    Args:
        merge_params: Mutable config dict with at least a ``keep_right_columns``
            key listing the desired right-hand columns.
        right_df: The DataFrame that will be used as the right side of the merge.
            Its columns *and* index names are treated as available sources.
        *args: Extra lists of column names (e.g. ``params:substation.group_columns``,
            ``params:county.group_columns``) whose entries should also be kept if
            present in ``right_df``.

    Returns:
        The mutated ``merge_params`` dict with ``keep_right_columns`` narrowed to
        the intersection of requested and available columns.
    """
    targets = merge_params["keep_right_columns"] + list(itertools.chain(*args))
    sources = right_df.columns.tolist() + right_df.index.names
    concats = list(set(targets).intersection(sources))
    merge_params["keep_right_columns"] = concats
    return merge_params


def merge_dataframes_node(
    left: pd.DataFrame | dd.DataFrame,
    right: pd.DataFrame,
    params: dict,
) -> pd.DataFrame | dd.DataFrame:
    """Left-join metadata from ``right`` onto ``left``, preserving ``left``'s index.

    Designed for the common Kedro pattern where a large fact table (``left``,
    possibly a Dask DataFrame) is enriched with a smaller lookup table (``right``,
    always in-memory pandas).  Only the columns listed in
    ``params["keep_right_columns"]`` are carried across; this avoids pulling
    unnecessary columns into the Dask graph.

    For Dask inputs, the join is executed partition-by-partition via
    ``map_partitions`` to avoid the expensive global ``set_index`` that a
    standard Dask merge would require.

    Args:
        left: The large fact DataFrame (pandas or Dask) whose index is preserved.
        right: The small metadata DataFrame to join onto ``left``.
        params: Configuration dict with the following keys:

            - **keep_right_columns** (``list[str]``): Columns from ``right`` (plus its
              index) to include in the output.  Must contain at least one entry.
            - **merge_kwargs** (``dict``): Keyword arguments forwarded verbatim to
              ``pandas.DataFrame.merge`` (e.g. ``on``, ``how``).

    Returns:
        Merged DataFrame with the same type and index as ``left`` and the selected
        columns from ``right`` appended.

    Raises:
        RuntimeError: If ``left`` is not a pandas or Dask DataFrame.
        RuntimeError: If ``right`` is not a pandas DataFrame.
        RuntimeError: If ``keep_right_columns`` is empty.
    """
    if not isinstance(left, pd.DataFrame | dd.DataFrame):
        raise RuntimeError("'left' must be a Pandas or Dask dataframe.")
    if not isinstance(right, pd.DataFrame):
        raise RuntimeError("'right' must be a Pandas dataframe.")
    is_dask = isinstance(left, dd.DataFrame)

    if not len(params["keep_right_columns"]) >= 1:
        raise RuntimeError(
            "At least one column must be kept from the 'right' dataframe"
        )

    mrg = right.reset_index()
    mrg = mrg.loc[:, params["keep_right_columns"]]

    def _merge_dataframe(
        left: pd.DataFrame, right: pd.DataFrame, mrg_kws: dict
    ) -> pd.DataFrame:
        orig_idx = left.index.names
        left = left.reset_index()

        merged = left.merge(right=right, **mrg_kws)

        if orig_idx != [None]:
            merged = merged.set_index(orig_idx)
        else:
            merged = merged.drop(columns=["index"])
        return merged

    if not is_dask:
        merged = _merge_dataframe(left=left, right=mrg, mrg_kws=params["merge_kwargs"])
    else:
        # For dask DataFrames, we need to preserve the index without expensive set_index operations
        merged = left.map_partitions(
            _merge_dataframe, right=mrg, mrg_kws=params["merge_kwargs"]
        )

    return merged


def merge_on_int_cols(
    left: dd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    **kwargs,
) -> dd.DataFrame:
    """Merge a pandas DataFrame onto a Dask DataFrame keyed on integer column(s).

    Dask's native merge on non-index columns requires a global ``set_index``,
    which is expensive.  This function avoids that cost by encoding up to two
    integer join columns into a single synthetic key (``id0 * max(id0) + id1``),
    performing a cheap index-based merge, then dropping the synthetic key.

    If the join column is already the Dask index, it is temporarily materialised
    as a column, merged, then restored as the index.

    Args:
        left: Dask DataFrame to enrich (large, partitioned).
        right: pandas DataFrame to join onto ``left`` (small, in-memory).
        on: Column name or list of up to two column names to join on.
        **kwargs: Additional keyword arguments forwarded to ``dd.merge``.

    Returns:
        Dask DataFrame with the same partitioning and index as ``left`` plus the
        columns from ``right``.

    Raises:
        NotImplementedError: If more than two join columns are supplied.
    """
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
    """Filter a DataFrame to rows where specified columns contain given values.

    Applies a sequence of ``isin`` filters, combining them with AND or OR logic
    as specified per column.  Optionally discards all columns not named in the
    filter spec.  GeoDataFrames are handled transparently: the geometry column
    is always retained regardless of ``keep_only_filter_cols``.

    Args:
        df: DataFrame or GeoDataFrame to filter.
        params: Configuration dict with the following keys:

            - **filters** (``dict[str, dict]``): Mapping of column name to a
              per-column filter spec.  Each spec may contain:

              - ``value_isin`` (``list``): Allowed values for this column.
              - ``invert`` (``bool``, optional): If ``True``, keep rows where the
                column is *not* in ``value_isin``.
              - ``joining_bool`` (``str``, optional): ``"AND"`` (default) or
                ``"OR"``; controls how this column's mask is combined with prior
                masks.

            - **keep_only_filter_cols** (``bool``): If ``True``, return only the
              columns named in ``filters`` (plus geometry for GeoDataFrames).

    Returns:
        Filtered DataFrame or GeoDataFrame.
    """
    df["filt"] = True
    keep_cols = []
    for col, filt in params["filters"].items():
        cur_ser = df[col].isin(filt["value_isin"])
        if "invert" in filt and filt["invert"]:
            cur_ser = ~cur_ser
        if "joining_bool" not in filt or filt["joining_bool"].upper() == "AND":
            df["filt"] &= cur_ser
        elif filt["joining_bool"].upper() == "OR":
            df["filt"] |= cur_ser
        keep_cols.append(col)
    if isinstance(df, gpd.GeoDataFrame):
        keep_cols += [df.geometry.name]
    if params["keep_only_filter_cols"]:
        filtered = df.loc[df["filt"], keep_cols]
    else:
        filtered = df.loc[df["filt"]]
        filtered = filtered.drop(columns="filt")
    return filtered


def get_basic_dtype_ser(ser: pd.Series) -> pd.Series:
    """Cast a Series to its equivalent non-nullable NumPy dtype (``int64`` or ``float64``).

    Pandas extension integer/float types (e.g. ``Int64``, ``Float32``) cannot be
    passed directly to Numba JIT functions.  This helper converts them to the
    corresponding plain NumPy dtype.

    Args:
        ser: Series with an integer or float dtype.

    Returns:
        Series cast to ``np.int64`` or ``np.float64``.

    Raises:
        RuntimeError: If ``ser`` has neither an integer nor a float dtype.
    """
    if pd.api.types.is_integer_dtype(ser):
        return ser.astype(np.int64)  # or 'int64' if appropriate
    elif pd.api.types.is_float_dtype(ser):
        return ser.astype(np.float64)
    else:
        raise RuntimeError("No available non-nullable dtype!")


class IndexIntegerizer:
    """Round-trip a (multi)index through a dense integer encoding for Numba compatibility.

    Numba JIT functions cannot operate on arbitrary pandas index values (strings,
    categoricals, MultiIndexes).  This class converts the index to a compact
    integer sequence via ``pandas.Index.factorize``, allowing the array to be
    passed to a JIT kernel, then restores the original index values afterwards.

    Typical usage::

        integerizer = IndexIntegerizer(int_col="veh_id_int")
        df_int = integerizer.integerize(df)
        result = numba_kernel(df_int)           # operates on integer index
        df_restored = integerizer.deintegerize(result)
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


def to_arrays(  # noqa: PLR0912
    df: pd.DataFrame, index: bool = True, column_dtypes=None, index_dtypes=None
) -> tuple:
    """Convert a DataFrame to arrays suitable for constructing a NumPy recarray.

    Extracts index and column data from ``df`` as a list of NumPy arrays,
    along with corresponding field names and dtype objects.  The returned
    triple can be passed directly to ``np.rec.fromarrays``.

    Args:
        df: DataFrame to convert.
        index: If ``True``, include the index as the first field(s); the
            field name is taken from the index label, or ``'index'`` if
            unlabelled.  Defaults to ``True``.
        column_dtypes: If a string or type, the data type to store all
            columns.  If a dictionary, a mapping of column names and
            zero-indexed positions to specific data types.  Defaults to
            ``None`` (infer from array dtype).
        index_dtypes: If a string or type, the data type to store all
            index levels.  If a dictionary, a mapping of index level names
            and zero-indexed positions to specific data types.  Applied
            only if ``index=True``.  Defaults to ``None``.

    Returns:
        Three-tuple ``(arrays, names, formats)`` where ``arrays`` is a
        list of NumPy arrays (one per included index level then column),
        ``names`` is the corresponding list of field name strings, and
        ``formats`` is the list of dtype objects.

    Raises:
        ValueError: If a ``dtype_mapping`` entry is not a type, a
            ``numpy.dtype``, or a string.
    """
    if index:
        ix_vals = [
            np.asarray(df.index.get_level_values(i)) for i in range(df.index.nlevels)
        ]

        arrays = ix_vals + [np.asarray(df.iloc[:, i]) for i in range(len(df.columns))]

        index_names = list(df.index.names)

        if isinstance(df.index, pd.MultiIndex):
            index_names = com.fill_missing_names(index_names)
        elif index_names[0] is None:
            index_names = ["index"]

        names = [str(name) for name in itertools.chain(index_names, df.columns)]
    else:
        arrays = [np.asarray(df.iloc[:, i]) for i in range(len(df.columns))]
        names = [str(c) for c in df.columns]
        index_names = []

    index_len = len(index_names)
    formats = []

    for i, v in enumerate(arrays):
        index_int = i

        # When the names and arrays are collected, we
        # first collect those in the DataFrame's index,
        # followed by those in its columns.
        #
        # Thus, the total length of the array is:
        # len(index_names) + len(DataFrame.columns).
        #
        # This check allows us to see whether we are
        # handling a name / array in the index or column.
        if index_int < index_len:
            dtype_mapping = index_dtypes
            name = index_names[index_int]
        else:
            index_int -= index_len
            dtype_mapping = column_dtypes
            name = df.columns[index_int]

        # We have a dictionary, so we get the data type
        # associated with the index or column (which can
        # be denoted by its name in the DataFrame or its
        # position in DataFrame's array of indices or
        # columns, whichever is applicable.
        if is_dict_like(dtype_mapping):
            if name in dtype_mapping:
                dtype_mapping = dtype_mapping[name]
            elif index_int in dtype_mapping:
                dtype_mapping = dtype_mapping[index_int]
            else:
                dtype_mapping = None

        # If no mapping can be found, use the array's
        # dtype attribute for formatting.
        #
        # A valid dtype must either be a type or
        # string naming a type.
        if dtype_mapping is None:
            formats.append(v.dtype)
        elif isinstance(dtype_mapping, (type, np.dtype, str)):  # noqa: UP038
            # error: Argument 1 to "append" of "list" has incompatible
            # type "Union[type, dtype[Any], str]"; expected "dtype[Any]"
            formats.append(dtype_mapping)  # type: ignore[arg-type]
        else:
            element = "row" if i < index_len else "column"
            msg = f"Invalid dtype {dtype_mapping} specified for {element} {name}"
            raise ValueError(msg)

    return (arrays, names, formats)


def categorize_columns(df: pd.DataFrame | DwellSet) -> pd.DataFrame | DwellSet:
    """Convert all ``object``-dtype columns to ``pandas.Categorical`` to reduce memory use.

    String columns stored as Python objects consume substantially more RAM than
    categorical columns with a small cardinality vocabulary (e.g. vehicle class,
    state code, hex cluster label).  This function is applied after loading or
    joining any DataFrame that may carry such columns.

    ``DwellSet`` inputs are handled transparently: only the underlying
    ``DwellSet.data`` DataFrame is modified; the wrapper is reconstructed via
    ``copy_without_data()`` so metadata is preserved.

    Args:
        df: DataFrame or DwellSet whose object-typed columns should be converted.

    Returns:
        Same type as ``df`` with object columns replaced by ``pd.Categorical``.
    """
    if isinstance(df, DwellSet):
        df_to_cat = df.data
    else:
        df_to_cat = df

    for col in df_to_cat.columns:
        if df_to_cat[col].dtype == np.dtype("O"):
            df_to_cat[col] = pd.Categorical(df_to_cat[col])

    if isinstance(df, DwellSet):
        out = df.copy_without_data()
        out.data = df_to_cat
    else:
        out = df_to_cat
    return out


def select_columns(
    df: pd.DataFrame | gpd.GeoDataFrame, params: dict
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Return ``df`` restricted to the columns listed in ``params["keep_cols"]``.

    For GeoDataFrames the geometry column is always appended to the selection
    so that spatial operations remain valid downstream.

    Args:
        df: DataFrame or GeoDataFrame to column-select.
        params: Configuration dict with the following key:

            - **keep_cols** (``str | list[str]``): Column name(s) to retain.

    Returns:
        DataFrame or GeoDataFrame containing only the requested columns (plus
        geometry for GeoDataFrames).
    """
    keep_cols = params["keep_cols"]
    if isinstance(keep_cols, str):
        keep_cols = [keep_cols]
    if isinstance(df, gpd.GeoDataFrame):
        keep_cols.append(df.geometry.name)
    return df.loc[:, keep_cols]
