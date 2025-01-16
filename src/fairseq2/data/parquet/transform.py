# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

from fairseq2.data.parquet.utils import hstack_pyarray_list, is_list_like
from fairseq2.logging import log


def build_uniform_list_column(
    array: Union[pa.Array, pa.ChunkedArray],
    length: int,
) -> pa.ChunkedArray:
    """
    Creates a ChunkedArray where each chunk is a single-list array containing
    all values in 'array', repeated 'length' times.

    For example, if array = [10, 11] (len(array) = 2), then for length=3
    the result is 3 identical ListArrays, each containing [10, 11].

    Args:
        array: The array to be repeated.
        length: The number of times to repeat the array.

    Returns:
        A ChunkedArray where each chunk is a single-list array containing the
        entire contents of `array`, repeated `length` times.
    """
    if len(array) == 0:  # Edge case: empty array
        single_list_array = pa.ListArray.from_arrays([0, 0], array)
    else:
        single_list_array = pa.ListArray.from_arrays([0, len(array)], array)
    repeated_arrays = [single_list_array] * length
    return pa.chunked_array(repeated_arrays)


def maybe_cast(
    arr: Union[pa.ChunkedArray, pa.Array],
    target_type: pa.DataType,
) -> Union[pa.ChunkedArray, pa.Array]:
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


def affix_list_column(
    table: pa.Table,
    column: str,
    *,
    prefix_array: pa.Array | None = None,
    suffix_array: pa.Array | None = None,
) -> pa.Array:
    """
    High-level helper that constructs and horizontally stacks prefix + original
    column + suffix for each row in a given list column.

    Args:
        table: The input pyarrow Table.
        column: Name of the list column to be modified.
        prefix_array: An array of prefix tokens to prepend.
        suffix_array: An array of suffix tokens to append.

    Returns:
        The new chunked array after prefix and suffix are applied.
    """
    prefix_extended = None
    suffix_extended = None
    target_type = table[column].type

    # Build repeated prefix/suffix for each row
    if prefix_array is not None:
        prefix_extended = build_uniform_list_column(prefix_array, len(table))
        prefix_extended = maybe_cast(prefix_extended, target_type)
    if suffix_array is not None:
        suffix_extended = build_uniform_list_column(suffix_array, len(table))
        suffix_extended = maybe_cast(suffix_extended, target_type)

    # Merge prefix, original column, and suffix
    arr2merge = []
    if prefix_extended is not None:
        arr2merge.append(prefix_extended)
    arr2merge.append(table[column])
    if suffix_extended is not None:
        arr2merge.append(suffix_extended)
    return hstack_pyarray_list(*arr2merge)


def replace_table_column(
    table: pa.Table,
    col_name: str,
    new_array: Union[pa.Array, pa.ChunkedArray],
) -> pa.Table:
    """
    Replaces an existing column in the Table with a new array.
    """
    col_idx = table.schema.get_field_index(col_name)
    return table.set_column(col_idx, col_name, new_array)


def prefix_and_suffix_one_list_column(
    table: pa.Table,
    column: str,
    prefix_array: pa.Array,
    suffix_array: pa.Array,
) -> pa.Table:
    """
    Adds a uniform prefix and suffix to each row of a list column.
    This is a simple wrapper around `affix_list_column` that updates
    the table in place.

    Args:
        table: The input pyarrow Table.
        column: Name of the list column to be modified.
        prefix_array: An array of prefix tokens to prepend.
        suffix_array: An array of suffix tokens to append.

    Returns:
        A new pyarrow Table with the modified list column.
    """
    new_array = affix_list_column(
        table, column, prefix_array=prefix_array, suffix_array=suffix_array
    )
    return replace_table_column(table, column, new_array)


def correct_paragraph_length(
    table: pa.Table,
    page_lens_column: str,
    len_break: int,
    len_prefix: int,
    len_suffix: int,
) -> pa.Table:
    """Adjusts paragraph-length arrays by adding specific offsets.

    When a prefix or suffix is injected into the text, the paragraph
    boundaries often need an extra offset. This function adds `len_break`
    to each element, plus `len_prefix - len_break` to the first element,
    and `len_suffix - len_break` to the last.

    Args:
        table: The input pyarrow Table.
        page_lens_column: Name of the column with paragraph-length arrays.
        len_break: Amount to add to every element except first/last.
        len_prefix: Additional offset for only the first element.
        len_suffix: Additional offset for only the last element.

    Returns:
        A new pyarrow Table with corrected paragraph-length arrays.
    """

    def _correct(row_lengths: List[int]) -> List[int]:
        # Shift entire row by len_break
        shifted = [val + len_break for val in row_lengths]
        # Then adjust first and last
        shifted[0] += len_prefix - len_break
        shifted[-1] += len_suffix - len_break
        return shifted

    # Convert to Python lists
    page_lengths = table[page_lens_column].to_pylist()
    corrected_data = [_correct(x) for x in page_lengths]

    # Rebuild into a PyArrow array
    corrected_lens = pa.array(corrected_data, type=pa.list_(pa.int32()))
    return replace_table_column(table, page_lens_column, corrected_lens)


def add_fragments_trace(table: pa.Table, fragment: pa.dataset.Fragment) -> pa.Table:
    """
    Adds a trace of the row groups and fragment ids to the table.

    Args:
        table: The input pyarrow Table.
        fragment: The fragment to trace.

    Returns:
        A new pyarrow Table with the trace added.
    """
    # Build a list-of-lists for row group IDs, one list per row.
    # Example: if table has 4 rows and fragment has row group IDs [0,1],
    # we'll end up with [[0,1], [0,1], [0,1], [0,1]] (length == 4).
    row_group_ids = [
        [int(rg.id) for rg in fragment.row_groups] for _ in range(len(table))
    ]

    # Explicitly specify that this is a list of int32, which also handles the case
    # where row_group_ids might be empty (e.g., zero row groups or zero rows in table).
    row_group_ids_array = pa.array(row_group_ids, type=pa.list_(pa.int32()))

    # Create a 1D int32 array for the index within the fragment
    fragment_index_array = pa.array(np.arange(len(table), dtype=np.int32))

    # Append both new columns
    table = table.append_column("__row_groups_ids", row_group_ids_array)
    table = table.append_column("__index_in_fragement", fragment_index_array)
    return table


def shuffle_table(table: pa.Table, random_state: np.random.RandomState) -> pa.Table:
    """
    Shuffles the rows of a table.

    Args:
        table: The input pyarrow Table.
        random_state: The random state to use for shuffling.

    Returns:
        A new pyarrow Table with shuffled rows.
    """
    permutation = pa.array(random_state.permutation(len(table)))
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
            (pl.col(column).list.eval(pl.col("").str.len_bytes()).list.min() >= min_len)
            & (
                pl.col(column).list.eval(pl.col("").str.len_bytes()).list.max()
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


def filter_rows_by_consistent_list_length(
    table: pa.Table, columns: List[str]
) -> pa.Table:
    """
    Keep only rows where all specified list-like columns have the same list length per row.

    If `columns` has length <= 1 or if any of the columns are not list-like,
    the table is returned unmodified (no filtering).

    For each column (after the first), rows are dropped if that column's list length
    differs from the reference column's list length. After each filtering step,
    the reference lengths are updated to match the newly filtered table.

    Args:
        table (pa.Table): A PyArrow Table.
        columns (List[str]): Names of the columns to check for consistent list lengths.
                             These columns should be list-like (ListArray or LargeListArray).

    Returns:
        pa.Table: A filtered PyArrow Table where all specified columns
                  have the same list length in each row.
    """
    # Quick exit if there's nothing to compare or columns are not all list-like
    if len(columns) <= 1:
        return table

    # Verify all specified columns are list-like
    if not all(is_list_like(table[col]) for col in columns):
        return table

    # Use the first column as our "reference" for lengths
    ref_lengths = pc.list_value_length(table[columns[0]])

    # Iterate over subsequent columns to confirm matching lengths
    for col in columns[1:]:
        col_lengths = pc.list_value_length(table[col])
        # same_lens[i] == True if row i has the same length in both columns
        same_lens = pc.equal(col_lengths, ref_lengths)

        # If all rows are consistent, move on
        if pc.all(same_lens).as_py():
            continue
        else:
            # Some rows differ => filter them out
            num_kept = pc.sum(same_lens).as_py()
            log.warning(
                f"Filtering rows with consistent list lengths among columns; "
                f"keeping {num_kept} out of {len(table)} rows."
            )
            table = table.filter(same_lens)
            # Also filter ref_lengths so it stays in sync
            ref_lengths = ref_lengths.filter(same_lens)

    return table
