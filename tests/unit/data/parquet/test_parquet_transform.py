# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from fairseq2.data.parquet.arrow_transform import (
    apply_filter,
    concat_table,
    filter_list_by_range,
    filter_strings_by_length,
    maybe_cast,
    repeat_list_column,
    replace_table_column,
    shuffle_table,
)


def test_repeat_list_column_empty() -> None:
    array = pa.array([], type=pa.int32())
    length = 3
    result = repeat_list_column(array, length)
    expected = pa.chunked_array([pa.ListArray.from_arrays([0, 0], array)] * length)
    assert result.equals(expected)


def test_repeat_list_column_non_empty() -> None:
    array = pa.array([10, 11], type=pa.int32())
    length = 2
    result = repeat_list_column(array, length)
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
