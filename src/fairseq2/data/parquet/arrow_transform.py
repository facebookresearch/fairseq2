# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Optional

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc


def is_list_like(arr: pa.ChunkedArray | pa.Array) -> bool:
    """
    Check if the array is a list or a large list.
    """
    return bool(pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type))


def _fix_list_offset(arr: pa.Array) -> pa.Array:
    """
    Recursively fixes list offset to 0, so that arr.offsets are always starts from 0
    and can be used easily downstream.
    """
    if not is_list_like(arr):
        return arr
    if arr.offset == 0 and arr.offsets[0].as_py() == 0:
        return arr

    new_values = _fix_list_offset(pc.list_flatten(arr))
    new_offsets = pc.subtract(arr.offsets, arr.offsets[0])

    return (
        pa.LargeListArray.from_arrays(new_offsets, new_values)
        if pa.types.is_large_list(arr.type)
        else pa.ListArray.from_arrays(new_offsets, new_values)
    )


def pyarrow_column_to_array(arg: pa.ChunkedArray | pa.Array) -> pa.Array:
    # see https://github.com/apache/arrow/issues/37318
    if isinstance(arg, pa.Array):
        return _fix_list_offset(arg)

    return _fix_list_offset(
        arg.chunk(0) if arg.num_chunks == 1 else arg.combine_chunks()
    )


def repeat_list_column(
    array: pa.Array | pa.ChunkedArray,
    length: int,
) -> pa.ChunkedArray:
    """
    >>> repeat_list_column(pa.array([1,2,3]), 2).to_pylist()
    [[1, 2, 3], [1, 2, 3]]
    """
    single_list_array = pa.ListArray.from_arrays([0, len(array)], array)
    repeated_arrays = [single_list_array] * length
    return pa.chunked_array(repeated_arrays)


def simple_array_to_nested(arr: pa.ChunkedArray | pa.Array) -> pa.Array:
    """
    >>> a = pa.array([1,2,3])
    >>> simple_array_to_nested(a).to_pylist()
    [[1], [2], [3]]
    """
    return pa.ListArray.from_arrays(
        pa.array(np.arange(len(arr) + 1, dtype=np.int32)), pyarrow_column_to_array(arr)
    )


def maybe_cast(
    arr: pa.ChunkedArray | pa.Array,
    target_type: pa.DataType,
) -> pa.ChunkedArray | pa.Array:
    """
    Casts a ChunkedArray to a target type if the current type does not match.

    Args:
        arr: The ChunkedArray to be cast.
        target_type: The target type to cast to.

    Returns:
        The casted ChunkedArray.
    """
    if not arr.type.equals(target_type):
        return arr.cast(target_type)
    return arr


def replace_table_column(
    table: pa.Table,
    col_name: str,
    new_array: pa.Array | pa.ChunkedArray,
) -> pa.Table:
    """
    Replaces an existing column in the Table with a new array.
    """
    col_idx = table.schema.get_field_index(col_name)
    return table.set_column(col_idx, col_name, new_array)


def shuffle_table(
    table: pa.Table, random_state: np.random.RandomState | None = None
) -> pa.Table:
    """
    Shuffles the rows of a table.

    Args:
        table: The input pyarrow Table.
        random_state: The random state to use for shuffling.

    Returns:
        A new pyarrow Table with shuffled rows.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    permutation: pa.Array = pa.array(random_state.permutation(len(table)))
    return table.take(permutation)


def apply_filter(
    table: pa.Table,
    filters: Optional[pa.dataset.Expression] = None,
    drop_null: bool = True,
) -> pa.Table:
    """
    Apply optional filtering to a PyArrow Table.

    Args:
        table (pa.Table): The input table.
        filters (Optional[pa.dataset.Expression]): A filter expression (e.g., ds.field('x') > 5).
        drop_null (bool): If True, drop rows where any column is null before applying `filters`.

    Returns:
        pa.Table: The resulting filtered table.
    """
    if drop_null:
        table = table.drop_null()
    if filters is not None:
        table = table.filter(filters)
    return table


def concat_table(tables: List[pa.Table], combine: bool = True) -> pa.Table:
    """
    Concatenates a list of PyArrow tables into one table.

    Args:
        tables (List[pa.Table]): Tables to concatenate.
        combine (bool): If True, combine chunks in the final table.

    Returns:
        pa.Table: A single concatenated table (optionally combined into fewer chunks).
    """
    result = pa.concat_tables(
        tables,
        promote_options="permissive",  # needed to get deal with empty segments
    )
    if combine:
        result = result.combine_chunks()
    return result


def filter_strings_by_length(
    table: pa.Table,
    column: str,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
) -> pa.Table:
    """
    Filter rows in a PyArrow Table based on the lengths of strings
    (in a list-of-strings column).

    Each row is included only if *all* strings within that row's list
    satisfy `min_len <= string_length <= max_len`.

    If every row passes, the original table is returned unchanged.

    Args:
        table (pa.Table): The input PyArrow table.
        column (str): Name of the list-of-strings column to inspect.
        min_len (int, optional): Minimum allowed string length (inclusive).
                                 Defaults to 0 if not provided.
        max_len (int, optional): Maximum allowed string length (inclusive).
                                 Defaults to a large integer if not provided.

    Returns:
        pa.Table: A filtered table containing only the rows
                  whose strings are within the specified length range.
    """
    if min_len is None and max_len is None:
        raise ValueError("At least one of min_len or max_len must be provided.")

    if min_len is None:
        min_len = 0

    if max_len is None:
        # set the upper bound
        max_len = 2**32

    # Convert just the target column to a Polars DataFrame
    # This assumes column is a list of strings.
    df_pl = pl.from_arrow(table.select([column]), rechunk=False)
    if not isinstance(df_pl, pl.DataFrame):
        raise TypeError("Polars conversion did not produce a DataFrame.")

    # Create a boolean mask:
    # (every string's length >= min_len) AND (every string's length <= max_len).
    # In Polars, we can evaluate each string's length within the list
    # and take the 'list.min()' and 'list.max()' as needed.
    filter_series = df_pl.with_columns(
        (
            (pl.col(column).list.eval(pl.col("").str.len_chars()).list.min() >= min_len)
            & (
                pl.col(column).list.eval(pl.col("").str.len_chars()).list.max()
                <= max_len
            )
        ).alias("mask")
    )["mask"].to_arrow()

    # If all rows pass, return the original table
    all_pass = pa.compute.all(filter_series).as_py()
    if all_pass:
        return table

    # Otherwise, filter the original table
    return table.filter(filter_series)


def filter_list_by_range(
    table: pa.Table,
    column: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> pa.Table:
    """
    Filter rows in a PyArrow Table based on the minimum and maximum
    values found in a list-of-numerical-values column.

    Each row is included if *all* values in its list satisfy:
        min_val <= value <= max_val

    If both min_val and max_val are None, returns the table unchanged.

    Args:
        table (pa.Table): The input table.
        column (str): The name of the column containing a list of numeric values.
        min_val (Optional[float]): The minimum allowed value (inclusive).
            Defaults to -∞ if not provided.
        max_val (Optional[float]): The maximum allowed value (inclusive).
            Defaults to +∞ if not provided.

    Returns:
        pa.Table: A filtered table containing only the rows
                  whose list-of-values are entirely in the [min_val, max_val] range.
    """
    if min_val is None and max_val is None:
        # No filtering needed
        return table

    if min_val is None:
        min_val = -float(np.inf)
    if max_val is None:
        max_val = float(np.inf)

    # Convert just the target column to a Polars DataFrame
    df_pl = pl.from_arrow(table.select([column]), rechunk=False)
    if not isinstance(df_pl, pl.DataFrame):
        raise TypeError("Polars conversion did not produce a DataFrame.")

    # Build a filter mask:
    # (list.max() <= max_val) AND (list.min() >= min_val)
    filter_series = df_pl.with_columns(
        (
            (pl.col(column).list.max() <= max_val)
            & (pl.col(column).list.min() >= min_val)
        ).alias("mask")
    )["mask"].to_arrow()

    # If all rows pass, return the table unchanged
    if pc.all(filter_series).as_py():
        return table

    # Otherwise, filter the original table
    return table.filter(filter_series)


def filter_list_with_min_max_length(
    table: pa.Table,
    columns: List[str],
    min_length: int,
    max_length: int,
) -> pa.Table:
    def _length_filter(column):
        filter_min = pc.greater_equal(pc.list_value_length(table[column]), min_length)
        filter_max = pc.less_equal(pc.list_value_length(table[column]), max_length)
        filter_ = pc.and_kleene(filter_min, filter_max)

        if pc.all(filter_).as_py():
            return table
        return table.filter(filter_)

    for column in columns:
        table = _length_filter(column)
    return table
