"""Tests for :mod:`laurel.utils.data`.

Guards against the regression fixed alongside ``describe_dwells.map_location_groups``:
feeding a plain pandas object into a Dask ``map_partitions`` call (as a kwarg, or
without first converting it to a Dask collection) embeds it as an unmanaged,
unspillable literal in *every* partition's task instead of one shared, spillable
graph node. Correctness-only tests can't catch this -- the merge result is
identical either way -- so these assert on the task graph directly.
"""

import cloudpickle
import dask.dataframe as dd
import pandas as pd
import pytest

from laurel.utils.data import merge_dataframes_node

N_ROWS = 4_000
N_PARTITIONS = 4


def _left() -> pd.DataFrame:
    return pd.DataFrame({"k": list(range(1000)) * 4, "v": range(N_ROWS)})


def _right() -> pd.DataFrame:
    # Padded to make duplication in the graph obvious against per-task overhead.
    return pd.DataFrame(
        {"k": range(1000), "w": [f"val_{i}_" + "pad" * 50 for i in range(1000)]}
    )


def _graph_task_bytes(ddf: dd.DataFrame, key_prefix: str) -> list[int]:
    graph = ddf.optimize().__dask_graph__()
    return [len(cloudpickle.dumps(v)) for k, v in graph.items() if k[0].startswith(key_prefix)]


def test_merge_dataframes_node_pandas_matches_dask():
    params = {"keep_right_columns": ["k", "w"], "merge_kwargs": {"on": "k", "how": "left"}}
    left, right = _left().rename_axis("idx"), _right()

    pandas_result = merge_dataframes_node(left, right, params)
    dask_result = merge_dataframes_node(
        dd.from_pandas(left, npartitions=N_PARTITIONS), right, params
    ).compute()

    pd.testing.assert_frame_equal(pandas_result, dask_result, check_dtype=False)


def test_merge_dataframes_node_dask_preserves_left_index():
    params = {"keep_right_columns": ["k", "w"], "merge_kwargs": {"on": "k", "how": "left"}}
    left = _left().rename_axis("idx")

    result = merge_dataframes_node(
        dd.from_pandas(left, npartitions=N_PARTITIONS), _right(), params
    ).compute()

    assert result.index.name == "idx"
    assert list(result.index) == list(left.index)


def test_merge_dataframes_node_does_not_duplicate_right_per_partition():
    """The ``right`` table must appear once in the graph, not once per partition.

    Regression guard for embedding ``right`` (or a derived subset of it) as an
    unmanaged literal via ``map_partitions(..., right=mrg, ...)`` -- passing a
    plain pandas object, or passing a Dask collection as a *kwarg* rather than
    positionally, both silently reintroduce the per-partition duplication this
    test is meant to catch.
    """
    params = {"keep_right_columns": ["k", "w"], "merge_kwargs": {"on": "k", "how": "left"}}
    right = _right()
    right_bytes = len(cloudpickle.dumps(right))

    out = merge_dataframes_node(dd.from_pandas(_left(), npartitions=N_PARTITIONS), right, params)

    per_partition_task_bytes = _graph_task_bytes(out, "_merge_dataframe")
    assert len(per_partition_task_bytes) == N_PARTITIONS
    # Each partition's task should be a small reference/closure, nowhere near
    # the size of the full right-hand table -- if `right` were duplicated per
    # partition, each of these would be roughly `right_bytes` on its own.
    assert max(per_partition_task_bytes) < right_bytes / 10


@pytest.mark.performance
def test_merge_dataframes_node_total_graph_size_scales_with_partitions_not_right_size():
    params = {"keep_right_columns": ["k", "w"], "merge_kwargs": {"on": "k", "how": "left"}}
    right = _right()
    right_bytes = len(cloudpickle.dumps(right))

    out = merge_dataframes_node(dd.from_pandas(_left(), npartitions=N_PARTITIONS), right, params)
    graph = out.optimize().__dask_graph__()
    total_bytes = sum(len(cloudpickle.dumps(v)) for v in graph.values())

    # If `right` were embedded once per partition, total graph size would be at
    # least N_PARTITIONS * right_bytes; sharing it once keeps the total well
    # under a 2x multiple of `right_bytes` plus the (small) partitioned data.
    assert total_bytes < 2 * right_bytes
