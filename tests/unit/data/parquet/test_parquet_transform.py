# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from fairseq2.data.parquet.transform import (
    add_fragments_trace,
    affix_list_column,
    apply_filter,
    build_uniform_list_column,
    concat_table,
    filter_list_by_range,
    filter_rows_by_consistent_list_length,
    filter_strings_by_length,
    maybe_cast,
    prefix_and_suffix_one_list_column,
    replace_table_column,
    shuffle_table,
)


def test_build_uniform_list_column_empty() -> None:
    array = pa.array([], type=pa.int32())
    length = 3
    result = build_uniform_list_column(array, length)
    expected = pa.chunked_array([pa.ListArray.from_arrays([0, 0], array)] * length)
    assert result.equals(expected)


def test_build_uniform_list_column_non_empty() -> None:
    array = pa.array([10, 11], type=pa.int32())
    length = 2
    result = build_uniform_list_column(array, length)
    single_list = pa.ListArray.from_arrays([0, 2], array)
    expected = pa.chunked_array([single_list] * length)
    assert result.equals(expected)


def test_maybe_cast_no_cast_needed() -> None:
    arr = pa.array([1, 2, 3], type=pa.int32())
    target_type = pa.int32()
    result = maybe_cast(arr, target_type)
    assert result.equals(arr)


def test_maybe_cast_required() -> None:
    arr = pa.array([1.0, 2.0, 3.0], type=pa.float64())
    target_type = pa.int32()
    result = maybe_cast(arr, target_type)
    expected = arr.cast(target_type)
    assert result.equals(expected)


def test_replace_table_column(sample_table: pa.Table) -> None:
    new_array = pa.array(
        [[10, 11, 12, 999], [13, 14, 999], [15, 999]], type=pa.list_(pa.int32())
    )
    expected_table = sample_table.drop(["tokens"]).append_column("tokens", new_array)
    result = replace_table_column(sample_table, "tokens", new_array)
    assert result.equals(expected_table)


def test_affix_list_column_no_prefix_suffix(sample_table: pa.Table) -> None:
    table = sample_table
    result = affix_list_column(table, "tokens", prefix_array=None, suffix_array=None)
    for i, v in enumerate(result):
        assert v.equals(table[1][i])


def test_affix_list_column_with_prefix(
    sample_table: pa.Table, prefix_array: pa.Array
) -> None:
    table = sample_table
    result = affix_list_column(
        table, "tokens", prefix_array=prefix_array, suffix_array=None
    )
    expected_tokens = pa.array(
        [
            [0, 1, 2, 3],
            [0, 4, 5],
            [0, 6],
        ],
        type=pa.list_(pa.int64()),
    )
    expected_table = replace_table_column(table, "tokens", expected_tokens)
    for i, v in enumerate(result):
        assert v.equals(expected_table[1][i])


def test_affix_list_column_with_suffix(
    sample_table: pa.Table, suffix_array: pa.Array
) -> None:
    table = sample_table
    result = affix_list_column(
        table, "tokens", prefix_array=None, suffix_array=suffix_array
    )
    expected_tokens = pa.array(
        [
            [1, 2, 3, 999],
            [4, 5, 999],
            [6, 999],
        ],
        type=pa.list_(pa.int64()),
    )
    expected_table = replace_table_column(table, "tokens", expected_tokens)
    for i, v in enumerate(result):
        assert v.equals(expected_table[1][i])


def test_affix_list_column_with_prefix_suffix(
    sample_table: pa.Table,
    prefix_array: pa.Array,
    suffix_array: pa.Array,
) -> None:
    result = affix_list_column(
        sample_table,
        "tokens",
        prefix_array=prefix_array,
        suffix_array=suffix_array,
    )
    expected_tokens = pa.array(
        [
            [0, 1, 2, 3, 999],
            [0, 4, 5, 999],
            [0, 6, 999],
        ],
        type=pa.list_(pa.int64()),
    )
    expected = replace_table_column(sample_table, "tokens", expected_tokens)
    for i, v in enumerate(result):
        assert v.equals(expected[1][i])


def test_prefix_suffix_list_column(
    sample_table: pa.Table,
    prefix_array: pa.Array,
    suffix_array: pa.Array,
) -> None:
    result = prefix_and_suffix_one_list_column(
        sample_table,
        "tokens",
        prefix_array=prefix_array,
        suffix_array=suffix_array,
    )
    expected_tokens = pa.array(
        [
            [0, 1, 2, 3, 999],
            [0, 4, 5, 999],
            [0, 6, 999],
        ],
        type=pa.list_(pa.int64()),
    )
    expected = replace_table_column(sample_table, "tokens", expected_tokens)
    assert result.equals(expected)


def test_correct_paragraph_length_basic(sample_table: pa.Table) -> None:
    # Create test data with shorter lines
    table = sample_table.append_column(
        "page_lens",
        pa.array([[3, 2, 1], [2, 1], [1]], type=pa.list_(pa.int32())),
    )

    result = correct_paragraph_length(
        table,
        "page_lens",
        len_break=1,
        len_prefix=2,
        len_suffix=3,
    )

    expected_lens = pa.array(
        [[5, 3, 4], [4, 4], [5]],
        type=pa.list_(pa.int32()),
    )
    expected = replace_table_column(table, "page_lens", expected_lens)
    assert result.equals(expected)


def test_correct_paragraph_length_empty() -> None:
    table = pa.Table.from_pydict(
        {"page_lens_column": pa.array([], type=pa.list_(pa.int32()))}
    )

    len_break = 1
    len_prefix = 2
    len_suffix = 3

    result = correct_paragraph_length(
        table, "page_lens_column", len_break, len_prefix, len_suffix
    )

    expected_page_lens = pa.array([], type=pa.list_(pa.int32()))
    expected_table = replace_table_column(table, "page_lens_column", expected_page_lens)
    assert result.equals(expected_table)


def test_correct_paragraph_length_single_element() -> None:
    table = pa.Table.from_pydict(
        {"page_lens_column": pa.array([[5]], type=pa.list_(pa.int32()))}
    )

    len_break = 2
    len_prefix = 3
    len_suffix = 4

    result = correct_paragraph_length(
        table, "page_lens_column", len_break, len_prefix, len_suffix
    )

    expected_page_lens = pa.array(
        [[5 + 2 + (3 - 2) + (4 - 2)]], type=pa.list_(pa.int32())
    )
    expected_table = replace_table_column(table, "page_lens_column", expected_page_lens)
    assert result.equals(expected_table)


class TestAddFragmentsTrace:
    def test_add_fragments_trace_basic(self) -> None:
        table = pa.table({"col1": [10, 20, 30, 40]})
        mock_fragment = MagicMock(spec=ds.Fragment)
        # Two row groups: IDs 0 and 1
        mock_fragment.row_groups = [MagicMock(id=0), MagicMock(id=1)]

        traced_table = add_fragments_trace(table, mock_fragment)

        assert traced_table.num_columns == 3
        assert traced_table.num_rows == 4

        # Check "__row_groups_ids" -> each row should have [0,1]
        rg_col = traced_table["__row_groups_ids"]
        rg_pylist = rg_col.to_pylist()
        expected = [[0, 1]] * 4
        assert rg_pylist == expected

        # Check "__index_in_fragement"
        idx_col = traced_table["__index_in_fragement"]
        idx_pylist = idx_col.to_pylist()
        assert idx_pylist == [0, 1, 2, 3]

    def test_add_fragments_trace_no_row_groups(self) -> None:
        table = pa.table({"col1": [1, 2, 3]})
        mock_fragment = MagicMock(spec=ds.Fragment)
        mock_fragment.row_groups = []  # no row groups

        traced_table = add_fragments_trace(table, mock_fragment)

        rg_pylist = traced_table["__row_groups_ids"].to_pylist()
        # Expect each row has an empty list
        assert rg_pylist == [[], [], []]

        idx_pylist = traced_table["__index_in_fragement"].to_pylist()
        assert idx_pylist == [0, 1, 2]

    def test_add_fragments_trace_empty_table(self) -> None:
        table = pa.table({"col1": []})  # 0 rows
        mock_fragment = MagicMock(spec=ds.Fragment)
        mock_fragment.row_groups = [MagicMock(id=99)]

        traced_table = add_fragments_trace(table, mock_fragment)

        assert traced_table.num_rows == 0
        assert traced_table.num_columns == 3

        rg_col = traced_table["__row_groups_ids"]
        assert len(rg_col) == 0  # no rows
        idx_col = traced_table["__index_in_fragement"]
        assert len(idx_col) == 0

    def test_add_fragments_trace_single_row(self) -> None:
        table = pa.table({"col1": [999]})
        mock_fragment = MagicMock(spec=ds.Fragment)
        mock_fragment.row_groups = [MagicMock(id=42)]

        traced_table = add_fragments_trace(table, mock_fragment)
        assert traced_table.num_rows == 1

        rg_pylist = traced_table["__row_groups_ids"].to_pylist()
        # Should be [[42]] (a list with a single sub-list containing 42)
        assert rg_pylist == [[42]]

        idx_pylist = traced_table["__index_in_fragement"].to_pylist()
        assert idx_pylist == [0]

    def test_add_fragments_trace_multiple_row_groups(self) -> None:
        table = pa.table({"col1": [10, 20]})
        mock_fragment = MagicMock(spec=ds.Fragment)
        # row group IDs: 100, 101, 102
        mock_fragment.row_groups = [
            MagicMock(id=100),
            MagicMock(id=101),
            MagicMock(id=102),
        ]

        traced_table = add_fragments_trace(table, mock_fragment)

        assert traced_table.num_rows == 2
        rg_pylist = traced_table["__row_groups_ids"].to_pylist()
        # Each of the 2 rows -> [100, 101, 102]
        assert rg_pylist == [[100, 101, 102], [100, 101, 102]]

        idx_pylist = traced_table["__index_in_fragement"].to_pylist()
        assert idx_pylist == [0, 1]


class TestShuffleTable:
    def test_shuffle_table_basic(self) -> None:
        table = pa.table({"col1": [1, 2, 3, 4], "col2": [10, 20, 30, 40]})
        # We'll seed the random generator for deterministic results
        random_state = np.random.RandomState(seed=42)

        shuffled = shuffle_table(table, random_state)

        # We should have the same schema, same length, but rows in a new order
        assert shuffled.schema == table.schema
        assert shuffled.num_rows == table.num_rows

        # Check that the contents are the same set of rows but permuted
        original_rows = set(tuple(row) for row in table.to_pylist())
        shuffled_rows = set(tuple(row) for row in shuffled.to_pylist())
        assert original_rows == shuffled_rows

    def test_shuffle_table_deterministic(self) -> None:
        table = pa.table({"col": list(range(5))})
        random_state = np.random.RandomState(seed=123)
        shuffled1 = shuffle_table(table, random_state)

        # Re-seed to replicate the same shuffle result
        random_state2 = np.random.RandomState(seed=123)
        shuffled2 = shuffle_table(table, random_state2)

        # Both shuffles should yield the same order
        assert shuffled1.equals(
            shuffled2
        ), "Shuffling with the same seed should produce identical results"

    def test_shuffle_table_empty(self) -> None:
        table = pa.table({"col1": []})
        random_state = np.random.RandomState(seed=0)
        shuffled = shuffle_table(table, random_state)

        assert shuffled.num_rows == 0
        assert shuffled.num_columns == 1
        assert shuffled.equals(
            table
        ), "Shuffling an empty table should return the same (empty) table"


class TestApplyFilter:
    def test_apply_filter_drop_null(self) -> None:
        table = pa.table({"col1": [1, None, 3], "col2": ["a", "b", None]})
        # drop_null=True should remove rows with ANY null value
        result = apply_filter(table, filters=None, drop_null=True)

        assert (
            result.num_rows == 1
        )  # only the row with (1, 'a') remains if both columns must be non-null
        pylist = result.to_pylist()
        assert pylist == [{"col1": 1, "col2": "a"}]

    def test_apply_filter_no_drop_null(self) -> None:
        table = pa.table({"col1": [1, None, 3], "col2": ["a", "b", None]})
        result = apply_filter(table, filters=None, drop_null=False)

        # With drop_null=False, no rows are removed yet
        assert result.num_rows == 3

    def test_apply_filter_with_expression(self) -> None:
        table = pa.table({"col1": [1, 2, 3, 4], "col2": [None, "x", "y", "z"]})
        # Drop null first => row 0 is gone
        # Then apply filters => keep rows where col1 > 2 => rows 2 & 3
        filters = ds.field("col1") > 2
        result = apply_filter(table, filters=filters, drop_null=True)

        # After drop_null, table had rows: [2,3,4], col2: ["x","y","z"] (index 1..3)
        # Then filter col1>2 => keep [3,4], col2=["y","z"]
        expected = pa.table({"col1": [3, 4], "col2": ["y", "z"]})
        assert result.equals(expected)

    def test_apply_filter_no_filters(self) -> None:
        table = pa.table({"col1": [1, 2], "col2": ["a", None]})
        # drop_null => remove row with None => row1
        result = apply_filter(table, None, drop_null=True)
        expected = pa.table({"col1": [1], "col2": ["a"]})
        assert result.equals(expected)

    def test_apply_filter_empty_table(self) -> None:
        empty_table = pa.table({"col1": []})
        # Even with drop_null or filters, still empty
        result = apply_filter(empty_table, drop_null=True)
        assert result.num_rows == 0
        assert result.equals(empty_table)


class TestConcatTable:
    def test_concat_table_basic(self) -> None:
        t1 = pa.table({"col1": [1, 2], "col2": ["a", "b"]})
        t2 = pa.table({"col1": [3, 4], "col2": ["c", "d"]})
        combined = concat_table([t1, t2], combine=True)

        # Expect a table with 4 rows
        assert combined.num_rows == 4
        # Combined chunks: With combine=True, ideally 1 chunk
        assert (
            len(combined.column(0).chunks) == 1
        ), "Should combine chunks into a single chunk"
        # Data check
        expected = pa.table({"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]})
        assert combined.equals(expected)

    def test_concat_table_no_combine(self) -> None:
        # If combine=False, the final table may have multiple chunks
        t1 = pa.table({"col1": [1, 2]})
        t2 = pa.table({"col1": [3, 4]})
        # We can artificially chunk them for demonstration
        t1 = t1.combine_chunks()  # ensure t1 is single-chunk
        t2 = t2.combine_chunks()  # ensure t2 is single-chunk

        result = concat_table([t1, t2], combine=False)
        # We expect 2 chunks
        col = result["col1"]
        assert len(col.chunks) == 2, "Should have kept 2 chunks because combine=False"
        assert result.to_pylist() == [
            {"col1": 1},
            {"col1": 2},
            {"col1": 3},
            {"col1": 4},
        ]

    def test_concat_table_empty_lists(self) -> None:
        empty1 = pa.table({"col1": []})
        empty2 = pa.table({"col1": []})
        result = concat_table([empty1, empty2])
        # Expect an empty table with at least the 'col1' column
        assert result.num_rows == 0
        assert result.num_columns == 1

    def test_concat_table_mixed_chunking(self) -> None:
        # Demonstrate different chunking scenarios
        arr1 = pa.chunked_array([[1, 2], [3]])  # 2 chunks
        arr2 = pa.chunked_array([[4]])  # 1 chunk
        t1 = pa.Table.from_arrays([arr1], ["col1"])
        t2 = pa.Table.from_arrays([arr2], ["col1"])
        result = concat_table([t1, t2], combine=True)

        # Expect a single chunk (because combine=True) with all data
        combined_col = result["col1"]
        assert len(combined_col.chunks) == 1
        assert result.to_pydict() == {"col1": [1, 2, 3, 4]}


class TestFilterStringsByLength:
    def test_filter_strings_by_length_all_pass(self) -> None:
        """
        Test that if all rows satisfy the min_len and max_len constraints,
        the original table is returned unchanged.
        """
        table = pa.table(
            {
                "id": [1, 2],
                "texts": [
                    ["hello", "world"],  # lengths = 5, 5
                    ["test", "data"],  # lengths = 4, 4
                ],
            }
        )
        # All strings have length between 3 and 5, so none should be filtered.
        result = filter_strings_by_length(table, column="texts", min_len=3, max_len=6)
        # Should return the same table
        assert result.equals(
            table
        ), "All rows pass => function should return the original table"

    def test_filter_strings_by_length_min_len(self) -> None:
        """
        Ensure that rows are dropped if any string is shorter than min_len.
        """
        table = pa.table(
            {
                "id": [1, 2, 3],
                "texts": [
                    ["hello", "hi"],  # lengths = 5, 2
                    ["foo", "bar"],  # lengths = 3, 3
                    ["tiny", "longish"],  # lengths = 4, 7
                ],
            }
        )
        # min_len=3 => row0 has a string "hi"=2 => should be removed
        # row1 => (3,3) => passes
        # row2 => (4,7) => passes
        result = filter_strings_by_length(table, "texts", min_len=3)
        expected = pa.table(
            {"id": [2, 3], "texts": [["foo", "bar"], ["tiny", "longish"]]}
        )
        assert result.equals(expected)

    def test_filter_strings_by_length_max_len(self) -> None:
        """
        Rows are dropped if any string exceeds max_len.
        """
        table = pa.table(
            {
                "id": [1, 2, 3],
                "texts": [
                    ["abc", "defg"],  # 3, 4
                    ["short", "excessively_long"],  # 5, 17
                    ["ten_chars", "exactly_10"],  # 9, 10
                ],
            }
        )
        # max_len=10 => row1 has a string "excessively_long"=17 => remove
        # row0 => (3,4) => passes
        # row2 => (9,10) => passes
        result = filter_strings_by_length(table, "texts", max_len=10)
        expected = pa.table(
            {"id": [1, 3], "texts": [["abc", "defg"], ["ten_chars", "exactly_10"]]}
        )
        assert result.equals(expected)

    def test_filter_strings_by_length_both_bounds(self) -> None:
        """
        Rows must have all string lengths in [min_len, max_len].
        """
        table = pa.table(
            {
                "id": [1, 2, 3],
                "texts": [
                    ["abc", "defg"],  # (3,4) => in [3..6]
                    ["hi", "hey"],  # (2,3) => "hi"=2 => out
                    ["abcdef", "zzz"],  # (6,3) => in
                ],
            }
        )
        result = filter_strings_by_length(table, "texts", min_len=3, max_len=6)
        expected = pa.table(
            {"id": [1, 3], "texts": [["abc", "defg"], ["abcdef", "zzz"]]}
        )
        assert result.equals(expected)

    def test_filter_strings_by_length_none_min_or_max(self) -> None:
        """
        If min_len is not specified, defaults to 0.
        If max_len is not specified, defaults to a large integer.
        """
        table = pa.table(
            {
                "id": [1, 2],
                "words": [
                    ["a", "b"],  # lengths=1,1 => allowed (min_len=0)
                    ["longtext", "x"],  # 8,1 => also allowed if no real upper bound
                ],
            }
        )
        # Only specify max_len => it effectively becomes huge => everything passes
        result = filter_strings_by_length(table, "words", max_len=1000)
        assert result.equals(table)

        # Now only specify min_len => everything passes because min_len=0 is default if omitted
        result2 = filter_strings_by_length(table, "words", min_len=1)
        # "a", "b" => length=1 => pass, "longtext"(8), "x"(1) => pass
        assert result2.equals(table)

    def test_filter_strings_by_length_no_rows(self) -> None:
        """
        If the table has 0 rows, the filter should return the table unchanged
        (still 0 rows).
        """
        table = pa.table(
            {
                "id": pa.array([], type=pa.int64()),
                "data": pa.array([], type=pa.list_(pa.string())),
            }
        )
        result = filter_strings_by_length(table, "data", min_len=1, max_len=5)
        assert result.equals(table)

    def test_filter_strings_by_length_no_min_no_max_raises(self) -> None:
        """
        If both min_len and max_len are None, the function should raise a ValueError.
        """
        table = pa.table({"col": [["abc"]]})
        with pytest.raises(ValueError):
            filter_strings_by_length(table, "col", min_len=None, max_len=None)


class TestFilterListByRange:
    def test_filter_list_by_range_no_bounds(self) -> None:
        """
        If both min_val and max_val are None, the function returns the original table unmodified.
        """
        table = pa.table({"id": [1, 2], "scores": [[0.5, 1.2], [3.4, 2.2]]})
        result = filter_list_by_range(table, "scores", None, None)
        assert result.equals(table), "No filtering should occur if both bounds are None"

    def test_filter_list_by_range_all_pass(self) -> None:
        """
        If all rows already satisfy the range, the original table is returned unchanged.
        """
        table = pa.table(
            {
                "id": [1, 2],
                "scores": [
                    [1.0, 2.0, 1.5],  # min=1.0, max=2.0 => in [1..3]
                    [2.5, 2.9],  # min=2.5, max=2.9 => in [1..3]
                ],
            }
        )
        result = filter_list_by_range(table, "scores", min_val=1, max_val=3)
        assert result.equals(table), "All rows pass => should return original"

    def test_filter_list_by_range_min_val(self) -> None:
        """
        Rows are dropped if any value in the list is below min_val.
        """
        table = pa.table(
            {
                "id": [1, 2, 3],
                "scores": [
                    [0.5, 2.0],  # min=0.5 => out if min_val=1
                    [1.2, 1.0],  # min=1.0 => OK if min_val=1
                    [2.1, 3.0],  # min=2.1 => definitely >=1
                ],
            }
        )
        # min_val=1 => row0 has 0.5 => out, row1 => (1.0,1.2), row2 => (2.1,3.0) => both pass
        result = filter_list_by_range(table, "scores", min_val=1, max_val=None)
        expected = pa.table({"id": [2, 3], "scores": [[1.2, 1.0], [2.1, 3.0]]})
        assert result.equals(expected)

    def test_filter_list_by_range_max_val(self) -> None:
        """
        Rows are dropped if any value in the list is above max_val.
        """
        table = pa.table(
            {
                "id": [1, 2, 3],
                "scores": [
                    [5.0, 9.9],  # max=9.9 => out if max_val=5
                    [4.9, 5.0],  # max=5 => OK
                    [1.2, 2.3],  # max=2.3 => definitely <=5
                ],
            }
        )
        # max_val=5 => row0's max=9.9 => out, row1=5 => pass, row2=2.3 => pass
        result = filter_list_by_range(table, "scores", min_val=None, max_val=5)
        expected = pa.table({"id": [2, 3], "scores": [[4.9, 5.0], [1.2, 2.3]]})
        assert result.equals(expected)

    def test_filter_list_by_range_both_bounds(self) -> None:
        """
        Rows must have all values in [min_val, max_val].
        """
        table = pa.table(
            {
                "id": [1, 2, 3],
                "scores": [
                    [2.0, 2.5],  # in [1..3]
                    [0.9, 2.0],  # 0.9 => out if min_val=1
                    [3.0, 2.9, 1.5],  # all in [1..3]
                ],
            }
        )
        result = filter_list_by_range(table, "scores", min_val=1, max_val=3)
        expected = pa.table({"id": [1, 3], "scores": [[2.0, 2.5], [3.0, 2.9, 1.5]]})
        assert result.equals(expected)

    def test_filter_list_by_range_no_rows(self) -> None:
        """
        If the table has 0 rows, the filter returns it unchanged.
        """
        empty_col = pa.array([], type=pa.list_(pa.float32()))
        table = pa.table({"id": [], "scores": empty_col})
        result = filter_list_by_range(table, "scores", min_val=1, max_val=5)
        assert result.num_rows == 0
        assert result.equals(table)

    def test_filter_list_by_range_all_pass_big_range(self) -> None:
        """
        If we give a huge range, effectively everything passes => returns original table.
        """
        table = pa.table({"id": [1, 2], "scores": [[-100.0, 50.0], [1e5, 1e6]]})
        # This is a giant range => all pass
        result = filter_list_by_range(table, "scores", min_val=-1e10, max_val=1e10)
        assert result.equals(table)


class TestFilterRowsByConsistentListLength:
    def test_no_columns(self) -> None:
        """
        If columns is empty or has only 1 column, the function should return the table unmodified.
        """
        table = pa.table({"col1": [[1, 2], [3, 4, 5]]})
        # 0 columns
        result = filter_rows_by_consistent_list_length(table, [])
        assert result.equals(table)

        # 1 column
        result_1col = filter_rows_by_consistent_list_length(table, ["col1"])
        assert result_1col.equals(table)

    def test_non_list_columns(self) -> None:
        """
        If any column is not list-like, the function returns the table unmodified.
        """
        table = pa.table({"col_list": [[1, 2], [3, 4]], "col_int": [100, 200]})
        result = filter_rows_by_consistent_list_length(table, ["col_list", "col_int"])
        # col_int is not list-like => returns unchanged
        assert result.equals(table)

    def test_all_consistent_lengths(self) -> None:
        """
        If all specified columns are list-like and have matching lengths in every row,
        the original table should be returned unchanged.
        """
        table = pa.table(
            {
                "list1": [[1, 2], [3, 4], [5, 6]],
                "list2": [[10, 20], [30, 40], [50, 60]],
                "other_col": [999, 888, 777],  # This won't be checked
            }
        )
        # list1 and list2 are each length 2 in every row
        columns = ["list1", "list2"]

        result = filter_rows_by_consistent_list_length(table, columns)
        assert result.equals(table), "All lengths match => no filtering"

    def test_inconsistent_lengths(self) -> None:
        """
        If some rows have differing lengths across columns, those rows should be filtered out.
        """
        table = pa.table(
            {
                "list1": [
                    [1, 2],  # length 2
                    [3, 4, 5],  # length 3
                    [],  # length 0
                ],
                "list2": [
                    [10, 20],  # length 2
                    [30, 40],  # length 2
                    [99],  # length 1
                ],
            }
        )
        # Row0 => list1=2, list2=2 => match
        # Row1 => list1=3, list2=2 => mismatch => filter out
        # Row2 => list1=0, list2=1 => mismatch => filter out
        columns = ["list1", "list2"]

        result = filter_rows_by_consistent_list_length(table, columns)
        expected = pa.table({"list1": [[1, 2]], "list2": [[10, 20]]})
        assert result.num_rows == 1
        assert result.equals(expected)

    def test_logging_warning_for_inconsistent(self) -> None:
        """
        Verify we log a warning message when rows get filtered out.
        """
        table = pa.table(
            {
                "list1": [[1, 2, 3], [4, 5], [6]],
                "list2": [[10, 20, 30], [40, 50, 60], [70, 80]],
            }
        )
        # Row1 => length(list1)=2, length(list2)=3 => mismatch => filtered
        # Row2 => length(list1)=1, length(list2)=2 => mismatch => filtered
        # Only Row0 => length3=length3 => keep

        columns = ["list1", "list2"]
        with patch("fairseq2.logging.log.warning") as mock_log:
            result = filter_rows_by_consistent_list_length(table, columns)
            # We expect 1 row kept => 2 filtered
            assert result.num_rows == 1
            assert mock_log.call_count > 0, "Should log at least one warning"

    def test_multiple_mismatches_across_more_columns(self) -> None:
        """
        Check a scenario with more than two columns.
        We filter step by step, updating the reference lengths after each filter.
        """
        table = pa.table(
            {
                "listA": [[1, 2], [3, 4, 5], [6, 7], [8, 9]],
                "listB": [[10, 20], [30, 40], [70, 80], [90, 100]],
                "listC": [[1000, 2000], [3000, 4000, 5000], [6000, 7000], [9000]],
            }
        )
        # Let's break down the mismatch:
        #   - For row0: listA=2, listB=2, listC=2 => all match
        #   - For row1: listA=3, listB=2, listC=3 => mismatch with listB
        #   - For row2: listA=2, listB=2, listC=2 => match
        #   - For row3: listA=2, listB=2, listC=1 => mismatch with listC
        # We keep row0, row2, filter out row1, row3.
        columns = ["listA", "listB", "listC"]
        result = filter_rows_by_consistent_list_length(table, columns)
        expected = pa.table(
            {
                "listA": [[1, 2], [6, 7]],
                "listB": [[10, 20], [70, 80]],
                "listC": [[1000, 2000], [6000, 7000]],
            }
        )
        assert result.equals(expected)

    def test_empty_table(self) -> None:
        """
        If the table is empty, there's nothing to filter. Return as is.
        """
        table = pa.table({"list1": pa.array([], type=pa.list_(pa.int64()))})
        columns = ["list1"]
        result = filter_rows_by_consistent_list_length(table, columns)
        assert result.num_rows == 0
        assert result.equals(table)
