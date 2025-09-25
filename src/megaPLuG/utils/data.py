import itertools
import logging
from typing import Self

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.core import common as com
from pandas.core.dtypes.common import is_dict_like

from megaPLuG.models.dwell_sets import DwellSet

logger = logging.getLogger(__name__)


def get_merge_params(
    merge_params: dict,
    right_df: pd.DataFrame,
    *args: list[list[str]],
) -> dict:
    """Update the params argument to the merge_datframes_node function."""
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
    """Merge two dataframes together as a Kedro node.

    This function assumes that the left dataframe is large, and that the right dataframe
    is adding some sort of metadata on to it.

    This function preserves the index of the left dataframe. It also allows you to
    select only a subset of columns from the right dataframe.
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


def to_arrays(  # noqa: PLR0912
    df: pd.DataFrame, index: bool = True, column_dtypes=None, index_dtypes=None
) -> np.rec.recarray:
    """
    Convert DataFrame to a NumPy record array.

    Index will be included as the first field of the record array if
    requested.

    Parameters
    ----------
    index : bool, default True
        Include index in resulting record array, stored in 'index'
        field or using the index label, if set.
    column_dtypes : str, type, dict, default None
        If a string or type, the data type to store all columns. If
        a dictionary, a mapping of column names and indices (zero-indexed)
        to specific data types.
    index_dtypes : str, type, dict, default None
        If a string or type, the data type to store all index levels. If
        a dictionary, a mapping of index level names and indices
        (zero-indexed) to specific data types.

        This mapping is applied only if `index=True`.

    Returns
    -------
    numpy.rec.recarray
        NumPy ndarray with the DataFrame labels as fields and each row
        of the DataFrame as entries.

    See Also
    --------
    DataFrame.from_records: Convert structured or record ndarray
        to DataFrame.
    numpy.rec.recarray: An ndarray that allows field access using
        attributes, analogous to typed columns in a
        spreadsheet.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [0.5, 0.75]},
    ...                   index=['a', 'b'])
    >>> df
       A     B
    a  1  0.50
    b  2  0.75
    >>> df.to_records()
    rec.array([('a', 1, 0.5 ), ('b', 2, 0.75)],
              dtype=[('index', 'O'), ('A', '<i8'), ('B', '<f8')])

    If the DataFrame index has no label then the recarray field name
    is set to 'index'. If the index has a label then this is used as the
    field name:

    >>> df.index = df.index.rename("I")
    >>> df.to_records()
    rec.array([('a', 1, 0.5 ), ('b', 2, 0.75)],
              dtype=[('I', 'O'), ('A', '<i8'), ('B', '<f8')])

    The index can be excluded from the record array:

    >>> df.to_records(index=False)
    rec.array([(1, 0.5 ), (2, 0.75)],
              dtype=[('A', '<i8'), ('B', '<f8')])

    Data types can be specified for the columns:

    >>> df.to_records(column_dtypes={"A": "int32"})
    rec.array([('a', 1, 0.5 ), ('b', 2, 0.75)],
              dtype=[('I', 'O'), ('A', '<i4'), ('B', '<f8')])

    As well as for the index:

    >>> df.to_records(index_dtypes="<S2")
    rec.array([(b'a', 1, 0.5 ), (b'b', 2, 0.75)],
              dtype=[('I', 'S2'), ('A', '<i8'), ('B', '<f8')])

    >>> index_dtypes = f"<S{df.index.str.len().max()}"
    >>> df.to_records(index_dtypes=index_dtypes)
    rec.array([(b'a', 1, 0.5 ), (b'b', 2, 0.75)],
              dtype=[('I', 'S1'), ('A', '<i8'), ('B', '<f8')])
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


def categorize_columns(df: pd.DataFrame | DwellSet) -> pd.DataFrame:
    """Categorize object-typed columns to save memory."""
    if isinstance(df, DwellSet):
        df_to_cat = df.data
    else:
        df_to_cat = df

    for col in df_to_cat.columns:
        if df_to_cat[col].dtype == np.dtype("O"):
            df_to_cat[col] = pd.Categorical(df_to_cat[col])
    return df
