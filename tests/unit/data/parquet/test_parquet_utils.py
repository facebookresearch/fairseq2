# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import tempfile
import unittest
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import torch
from pyarrow.dataset import Fragment

from fairseq2.data.parquet.utils import (
    _fix_list_offset,
    add_partitioning_values,
    compute_length_splits,
    compute_rows_length,
    get_dataset_fragments,
    hstack_pyarray_list,
    is_list_like,
    load_one_fragment,
    pyarrow_column_to_array,
    pyarrow_cpu,
    pyarrow_table_to_torch_dict,
    pyarrow_to_torch_tensor,
    split_fragment_in_row_groups,
    torch_random_seed,
)


def test_nested_text_conversion() -> None:
    nested_input = pa.array([[["abc", "efg"]]])
    tt = pa.Table.from_pydict({"nested_text": nested_input})
    with pytest.raises(NotImplementedError):
        pyarrow_table_to_torch_dict(tt)


def test_is_list_like() -> None:
    # Test regular list array
    list_arr = pa.array([[1, 2], [3, 4]])
    assert is_list_like(list_arr) is True

    # Test large list array
    large_list_arr = pa.array([[1] * 1000] * 1000)
    assert is_list_like(large_list_arr) is True

    # Test non-list array
    regular_arr = pa.array([1, 2, 3])
    assert is_list_like(regular_arr) is False


def test_fix_list_offset() -> None:
    # Test array with no offset
    arr = pa.array([[1, 2], [3, 4]])
    fixed = _fix_list_offset(arr)
    assert arr.equals(fixed)

    # Test array with offset
    data = [[1, 2], [3, 4], [5, 6]]
    arr_with_offset = pa.array(data).slice(1, 2)  # offset of 1
    fixed = _fix_list_offset(arr_with_offset)
    assert fixed.offset == 0
    assert fixed.to_pylist() == data[1:3]

    # Test nested list array with offset
    nested_data = [[[1], [2]], [[3], [4]], [[5], [6]]]
    nested_arr = pa.array(nested_data).slice(1, 2)
    fixed = _fix_list_offset(nested_arr)
    assert fixed.offset == 0
    assert fixed.to_pylist() == nested_data[1:3]


def test_pyarrow_column_to_array() -> None:
    # Test with regular Array
    arr = pa.array([[1, 2], [3, 4]])
    result = pyarrow_column_to_array(arr)
    assert result.equals(arr)

    # Test with ChunkedArray (single chunk)
    chunked = pa.chunked_array([arr])
    result = pyarrow_column_to_array(chunked)
    assert result.equals(arr)

    # Test with ChunkedArray (multiple chunks)
    arr1 = pa.array([[1, 2]])
    arr2 = pa.array([[3, 4]])
    chunked = pa.chunked_array([arr1, arr2])
    result = pyarrow_column_to_array(chunked)
    assert result.to_pylist() == [[1, 2], [3, 4]]


def test_hstack_pyarray_list() -> None:
    # Test with simple lists
    a = pa.array([[1], [2, 3], [5], []])
    b = pa.array([[-1, -3], [-11], [], [22]])
    result = hstack_pyarray_list(a, b)
    expected = [[1, -1, -3], [2, 3, -11], [5], [22]]
    assert result.to_pylist() == expected

    # Test with nested lists
    data1 = [[[1, 2]], [[3, 4]], [[5, 6]]]
    data2 = [[[7, 8]], [[9, 10]], [[11, 12]]]
    arr1 = pa.array(data1)
    arr2 = pa.array(data2)
    result = hstack_pyarray_list(arr1, arr2)
    expected = [
        [[1, 2], [7, 8]],  # type: ignore
        [[3, 4], [9, 10]],  # type: ignore
        [[5, 6], [11, 12]],  # type: ignore
    ]
    assert result.to_pylist() == expected

    # Test with arrays of different types (regular and large)
    regular = pa.array([[1], [2]])
    large = pa.array([[1] * 1000] * 2)
    result = hstack_pyarray_list(regular, large)
    assert len(result) == 2


def test_hstack_pyarray_list_errors() -> None:
    # Test arrays of different lengths
    a = pa.array([[1], [2]])
    b = pa.array([[3]])
    with pytest.raises(ValueError):
        hstack_pyarray_list(a, b)

    # Test with non-list arrays
    regular_arr = pa.array([1, 2, 3])
    list_arr = pa.array([[1], [2], [3]])
    with pytest.raises(ValueError):
        hstack_pyarray_list(regular_arr, list_arr)


def test_pyarrow_to_torch_tensor_primitive() -> None:
    # Test primitive types
    arr = pa.array([1, 2, 3, 4])
    result = pyarrow_to_torch_tensor(arr)
    assert isinstance(result, torch.Tensor)
    assert torch.all(result == torch.tensor([1, 2, 3, 4]))

    # Test float array
    arr = pa.array([1.0, 2.0, 3.0])
    result = pyarrow_to_torch_tensor(arr)
    assert isinstance(result, torch.Tensor)
    assert torch.all(result == torch.tensor([1.0, 2.0, 3.0]))


def test_pyarrow_to_torch_tensor_string() -> None:
    # Test string arrays
    arr = pa.array(["hello", "world"])
    result = pyarrow_to_torch_tensor(arr)
    assert isinstance(result, list)
    assert result == ["hello", "world"]


def test_pyarrow_to_torch_tensor_list() -> None:
    # Test list of primitives
    arr = pa.array([[1, 2], [3, 4]])
    result = pyarrow_to_torch_tensor(arr)
    assert isinstance(result, list)
    assert len(result) == 2
    assert torch.all(result[0] == torch.tensor([1, 2]))
    assert torch.all(result[1] == torch.tensor([3, 4]))

    # Test empty list
    arr = pa.array([[]], type=pa.list_(pa.int64()))
    result = pyarrow_to_torch_tensor(arr)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], torch.Tensor)
    assert result[0].numel() == 0


def test_pyarrow_to_torch_tensor_fixed_size_list() -> None:
    # Test fixed size list
    data = [[1, 2], [3, 4], [5, 6]]
    arr = pa.array(data, type=pa.list_(pa.int64(), 2))
    result = pyarrow_to_torch_tensor(arr)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 2)
    assert torch.all(result == torch.tensor([[1, 2], [3, 4], [5, 6]]))


def test_pyarrow_to_torch_tensor_struct() -> None:
    # Test struct array
    struct_data = [{"x": 1, "y": 2.0}, {"x": 3, "y": 4.0}]
    arr = pa.array(struct_data)
    result = pyarrow_to_torch_tensor(arr)
    assert isinstance(result, dict)
    assert "x" in result and "y" in result
    assert torch.all(result["x"] == torch.tensor([1, 3]))
    assert torch.all(result["y"] == torch.tensor([2.0, 4.0]))


def test_pyarrow_to_torch_tensor_dictionary() -> None:
    # Test dictionary encoded array
    indices = pa.array([0, 1, 0])
    dictionary = pa.array(["a", "b"])
    arr = pa.DictionaryArray.from_arrays(indices, dictionary)
    result = pyarrow_to_torch_tensor(arr)
    assert isinstance(result, list)
    assert result == ["a", "b", "a"]


def test_pyarrow_to_torch_tensor_nested_list() -> None:
    # Test nested list with fixed size inner lists
    data = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]
    arr = pa.array(data)
    with pytest.raises(NotImplementedError):
        pyarrow_to_torch_tensor(arr)


def test_pyarrow_to_torch_tensor_null_values() -> None:
    # Test that null values raise ValueError
    arr = pa.array([1, None, 3])
    with pytest.raises(
        ValueError, match="to torch conversion does not support null values"
    ):
        pyarrow_to_torch_tensor(arr)


def test_pyarrow_cpu_context_manager() -> None:
    original_cpu_count = pa.cpu_count()
    original_io_thread_count = pa.io_thread_count()

    new_cpu_count = max(1, original_cpu_count - 1)  # Ensure at least 1

    with pyarrow_cpu(new_cpu_count):
        assert pa.cpu_count() == new_cpu_count
        assert pa.io_thread_count() == new_cpu_count

    # After context, original settings should be restored
    assert pa.cpu_count() == original_cpu_count
    assert pa.io_thread_count() == original_io_thread_count


def test_pyarrow_cpu_context_manager_exception() -> None:
    original_cpu_count = pa.cpu_count()
    original_io_thread_count = pa.io_thread_count()

    new_cpu_count = max(1, original_cpu_count - 1)  # Ensure at least 1

    try:
        with pyarrow_cpu(new_cpu_count):
            assert pa.cpu_count() == new_cpu_count
            assert pa.io_thread_count() == new_cpu_count
            raise RuntimeError("Test exception within context manager")
    except RuntimeError as e:
        assert str(e) == "Test exception within context manager"

    # After exception, original settings should be restored
    assert pa.cpu_count() == original_cpu_count
    assert pa.io_thread_count() == original_io_thread_count


def test_torch_random_seed_context_manager_with_seed() -> None:
    seed = 42
    with torch_random_seed(seed):
        torch.manual_seed(seed)
        tensor1 = torch.rand(3)

    # After context, RNG state should be restored
    torch.manual_seed(0)  # Change seed to differentiate
    tensor2 = torch.rand(3)

    # Verify that tensor1 is reproducible
    torch.manual_seed(seed)
    tensor1_repro = torch.rand(3)
    assert torch.all(tensor1 == tensor1_repro)

    # Verify that tensor2 is different
    torch.manual_seed(0)
    tensor2_repro = torch.rand(3)
    assert torch.all(tensor2 == tensor2_repro)


def test_torch_random_seed_context_manager_no_seed() -> None:
    original_rng_state = torch.get_rng_state()

    with torch_random_seed(None):
        # RNG state should remain unchanged
        assert torch.get_rng_state().equal(original_rng_state)
        tensor1 = torch.rand(3)

    # After context, RNG state should remain unchanged
    tensor2 = torch.rand(3)

    # Verify that tensor1 and tensor2 are part of the same RNG sequence
    torch.set_rng_state(original_rng_state)
    tensor1_repro = torch.rand(3)
    assert torch.all(tensor1 == tensor1_repro)

    tensor2_repro = torch.rand(3)
    assert torch.all(tensor2 == tensor2_repro)


def create_sample_parquet_dataset(
    base_dir: str,
    num_files: int = 3,
    num_row_groups: int = 2,
    partition_key: Optional[str] = None,
    partition_value: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> List[pa.Table]:
    """
    Create a sample Parquet dataset with optional partitioning.

    This function generates multiple Parquet files (by default, 3),
    each containing a small PyArrow table. The number of row groups
    is also configurable. Optionally, a partition column (key/value)
    can be added to each table to simulate partitioning scenarios.
    A random seed may be specified for reproducible random data.

    Args:
        base_dir (str): Directory in which to create the Parquet files.
        num_files (int, optional): Number of Parquet files to generate.
            Defaults to 3.
        num_row_groups (int, optional): Number of row groups per file.
            Defaults to 2.
        partition_key (str, optional): Column name for partitioning (if any).
            Defaults to None.
        partition_value (str, optional): Partition value for the added partition key.
            Defaults to None.
        random_seed (int, optional): Seed for NumPy's random generator.
            If not provided, data will be randomized each time.

    Returns:
        List[pa.Table]: List of PyArrow tables that were written to disk.

    Raises:
        ValueError: If the number of row groups is zero or negative.

    Example:
        >>> create_sample_parquet_dataset("sample_data", num_files=2, num_row_groups=1)
        [pyarrow.Table, pyarrow.Table]
    """
    if num_row_groups <= 0:
        raise ValueError("num_row_groups must be a positive integer.")

    if random_seed is not None:
        np.random.seed(random_seed)

    os.makedirs(base_dir, exist_ok=True)
    tables = []

    # Decide how many rows go into each row group.
    # For simplicity, assume total rows is 10, then divide by num_row_groups.
    # Adjust as needed for your use case.
    row_group_size = np.ceil(10 // num_row_groups)

    for i in range(num_files):
        # Create a simple dataset of 'id' and 'value'
        start_idx = i * 10
        end_idx = (i + 1) * 10
        data = {
            "id": np.arange(start_idx, end_idx),
            "value": np.random.rand(10),
        }

        table = pa.table(data)

        # Add partition column if both partition_key and partition_value are provided
        if partition_key and partition_value:
            partition_array = pa.array([partition_value] * 10)
            table = table.append_column(partition_key, partition_array)

        # Write the table to a Parquet file
        pq_file_path = os.path.join(base_dir, f"data_{i}.parquet")
        pq.write_table(table, pq_file_path, row_group_size=row_group_size)

        # Collect the in-memory table for return
        tables.append(table)

    return tables


def test_get_dataset_fragments() -> None:
    """
    Test the get_dataset_fragments function to ensure it returns the correct fragments based on filters.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample dataset with 3 Parquet files
        create_sample_parquet_dataset(tmpdir, num_files=3, num_row_groups=2)

        # Create a ParquetDataset
        dataset = pq.ParquetDataset(tmpdir)

        # Define a filter to select rows where id >= 10
        filter_expr = ds.field("id") >= 10

        # Call the function
        fragments = get_dataset_fragments(dataset, filter_expr)

        # Verify that fragments are returned
        assert isinstance(fragments, list)
        assert (
            len(fragments) == 3
        )  # Since each file has 2 row groups, total fragments should match

        for fragment in fragments:
            assert isinstance(fragment, Fragment)


def test_add_partitioning_values() -> None:
    """
    Test the add_partitioning_values function to ensure it correctly adds partitioning columns to the table.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define partition key and value
        partition_key = "country"
        partition_value = "US"

        # Create a sample dataset with partitioning
        create_sample_parquet_dataset(
            tmpdir,
            num_files=2,
            num_row_groups=1,
            partition_key=partition_key,
            partition_value=partition_value,
        )

        # Create a ParquetDataset with partitioning
        dataset = pq.ParquetDataset(tmpdir)

        # Get fragments
        fragments = get_dataset_fragments(dataset, ds.field("id") >= 0)
        assert len(fragments) == 2

        # Load a fragment
        fragment = fragments[0]
        table = fragment.to_table()

        # Add partitioning values
        updated_table = add_partitioning_values(table, fragment, columns=None)

        # Verify that the partitioning column is added
        assert partition_key in updated_table.column_names
        partition_column = updated_table.column(partition_key).to_pylist()
        assert all(val == partition_value for val in partition_column)


def test_add_partitioning_values_no_partition_columns() -> None:
    """
    Test the add_partitioning_values function when there are no partitioning columns to add.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample dataset without partitioning
        create_sample_parquet_dataset(tmpdir, num_files=2, num_row_groups=1)

        # Create a ParquetDataset without partitioning
        dataset = pq.ParquetDataset(tmpdir)

        # Get fragments
        fragments = get_dataset_fragments(dataset, ds.field("id") >= 0)
        assert len(fragments) == 2

        # Load a fragment
        fragment = fragments[0]
        table = fragment.to_table()

        # Add partitioning values (should not add any columns)
        updated_table = add_partitioning_values(table, fragment, columns=None)

        # Verify that no new columns are added
        assert updated_table.column_names == table.column_names


def test_split_fragment_in_row_groups_no_row_groups() -> None:
    """
    Test the split_fragment_in_row_groups function with a fragment that has no row groups.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample Parquet file with no row groups (this is atypical, but for testing)
        data = {
            "id": np.arange(10),
            "value": np.random.rand(10),
        }
        table = pa.Table.from_pandas(pd.DataFrame(data))
        pq_file = os.path.join(tmpdir, "data.parquet")
        pq.write_table(table, pq_file)

        # Create a ParquetDataset
        dataset = pq.ParquetDataset(tmpdir)

        # Get fragments
        fragments = get_dataset_fragments(dataset, ds.field("id") >= 0)
        assert len(fragments) == 1
        fragment = fragments[0]

        # Split the fragment into row groups
        split_fragments = split_fragment_in_row_groups(fragment)

        # Verify that at least one fragment is returned
        assert len(split_fragments) >= 1
        assert all(isinstance(frag, Fragment) for frag in split_fragments)


class TestLoadOneFragment(unittest.TestCase):
    @patch("fairseq2.data.parquet.utils.add_partitioning_values")
    def test_load_one_fragment_with_no_columns(
        self, mock_add_partitioning_values: MagicMock
    ) -> None:
        """Test loading fragment when columns=None (should load all)."""
        # Setup a mock fragment
        mock_fragment = MagicMock(spec=ds.Fragment)
        mock_fragment.physical_schema.names = ["col1", "col2", "col3"]

        # Mock the table returned by fragment.to_table
        mock_table = MagicMock(spec=pa.Table)
        mock_fragment.to_table.return_value = mock_table

        # Mock the add_partitioning_values return
        mock_add_partitioning_values.return_value = mock_table

        # Call the function under test
        result_table = load_one_fragment(mock_fragment, columns=None)

        # Assertions
        mock_fragment.to_table.assert_called_once_with(columns=None, use_threads=False)
        mock_add_partitioning_values.assert_called_once_with(
            mock_table, mock_fragment, None
        )
        self.assertEqual(result_table, mock_table)

    @patch("fairseq2.data.parquet.utils.add_partitioning_values")
    def test_load_one_fragment_with_subset_of_columns(
        self, mock_add_partitioning_values: MagicMock
    ) -> None:
        """Test loading fragment when columns are specified (including some that don't exist)."""
        # Setup a mock fragment
        mock_fragment = MagicMock(spec=ds.Fragment)
        mock_fragment.physical_schema.names = ["col1", "col2", "col3"]

        # Mock the table returned by fragment.to_table
        mock_table = MagicMock(spec=pa.Table)
        mock_fragment.to_table.return_value = mock_table

        # Mock the add_partitioning_values return
        mock_add_partitioning_values.return_value = mock_table

        # Suppose we request columns "col2" and "colX" (which does not exist)
        requested_columns = ["col2", "colX"]
        result_table = load_one_fragment(mock_fragment, columns=requested_columns)

        # Only "col2" should be passed to fragment.to_table, since "colX" doesn't exist
        mock_fragment.to_table.assert_called_once_with(
            columns=["col2"], use_threads=False
        )
        mock_add_partitioning_values.assert_called_once_with(
            mock_table, mock_fragment, requested_columns
        )
        self.assertEqual(result_table, mock_table)

    @patch("fairseq2.data.parquet.utils.add_partitioning_values")
    def test_load_one_fragment_empty_physical_schema(
        self, mock_add_partitioning_values: MagicMock
    ) -> None:
        """Test behavior when the fragment has an empty physical schema."""
        # Setup a mock fragment with empty schema
        mock_fragment = MagicMock(spec=ds.Fragment)
        mock_fragment.physical_schema.names = []

        # Mock the table returned by fragment.to_table
        mock_table = MagicMock(spec=pa.Table)
        mock_fragment.to_table.return_value = mock_table

        # Mock the add_partitioning_values return
        mock_add_partitioning_values.return_value = mock_table

        # Request columns that don't exist
        requested_columns = ["col1", "col2"]
        result_table = load_one_fragment(mock_fragment, columns=requested_columns)

        # The fragment.to_table should receive columns=None or an empty list
        mock_fragment.to_table.assert_called_once_with(columns=[], use_threads=False)
        mock_add_partitioning_values.assert_called_once_with(
            mock_table, mock_fragment, requested_columns
        )
        self.assertEqual(result_table, mock_table)


@pytest.mark.parametrize(
    "length_col,max_tokens,order_by_length,drop_long_sample,expected_splits",
    [
        # ---------------------------------------------------------------------
        # Test 1: Simple case with all samples fitting easily if sorted
        (
            np.array([1, 2, 3, 4, 2, 1], dtype=np.int32),
            8,
            True,
            True,
            [
                # Explanation:
                # Sorted lengths = [1, 1, 2, 2, 3, 4]
                # Indices might be [0, 5, 1, 4, 2, 3] (depending on how they were sorted)
                # Let's see how the function would group them based on max_tokens=8
                # We'll verify after we see the actual result in the test.
                # This test just ensures the function completes without
                # rejecting any items, because all fit in some chunk.
                # We'll do a partial verification on the shape (# of splits) and total coverage.
            ],
        ),
        # ---------------------------------------------------------------------
        # Test 2: No sorting
        (
            np.array([1, 5, 2], dtype=np.int32),
            6,
            False,
            True,
            [
                # Without sorting, we process in order: [1, 5, 2]
                # - First item (1) sets chunk size of 1 => can we add 5?
                #   new max_len=5, 2 items => 5*2=10 > 6 => we close the first chunk with [1].
                # - Next chunk starts at 5 => can we add 2?
                #   new max_len=5, 2 items => 10 > 6 => chunk will be just [5].
                # - Next chunk is [2].
            ],
        ),
        # ---------------------------------------------------------------------
        # Test 3: Large items that must be dropped
        (
            np.array([2, 10, 4], dtype=np.int32),
            5,
            True,
            True,
            [
                # 10 is bigger than max_tokens, so it's dropped entirely.
                # Only [2,4] remain in sorted order => [2,4].
                # Next, 2 => chunk with 2 is okay; max_len=2 => 2*1=2 <=5 => try adding 4
                # new max_len=4 => 4*2=8 >5 => so chunk is [2], next chunk is [4]
            ],
        ),
        # ---------------------------------------------------------------------
        # Test 4: Large items that must be kept as singletons
        (
            np.array([2, 10, 4], dtype=np.int32),
            5,
            True,
            False,
            [
                # Now we keep the large item (10) in a separate chunk
                # Based on the logic:
                # - 10 is bigger than max_tokens => goes into big_elements_inds
                # - For the small ones [2,4], same logic as above => chunk for [2], chunk for [4].
                # - Add the big item (10) at the end as a separate chunk => [10].
            ],
        ),
    ],
)
def test_compute_length_splits_parametrized(
    length_col: np.ndarray[int, np.dtype[np.int32]],
    max_tokens: int,
    order_by_length: bool,
    drop_long_sample: bool,
    expected_splits: List[np.ndarray[int, np.dtype[np.int32]]],
) -> None:
    """
    Parametrized test that checks compute_length_splits for different scenarios.
    """
    result = compute_length_splits(
        length_col=length_col,
        max_tokens=max_tokens,
        order_by_length=order_by_length,
        drop_long_sample=drop_long_sample,
    )

    # Basic checks:
    # 1) All items that are supposed to remain are included exactly once.
    # 2) The shape (i.e., how many splits) can be tested if we know it exactly.
    # 3) For samples that are dropped, verify they are not in the result.

    # Flatten the resulting splits
    merged_indices = np.concatenate(result, axis=0) if len(result) else []  # type: ignore
    unique_merged_indices = np.unique(merged_indices)

    # Which indices are "allowed"? (If dropping, exclude those > max_tokens)
    allowed_mask = length_col <= max_tokens
    allowed_indices = np.where(allowed_mask)[0]

    if drop_long_sample:
        # Ensure no big sample is in the result
        assert not any(length_col[idx] > max_tokens for idx in merged_indices)

    # Check that everything "allowed" is present in the final splits
    # and nothing else (if dropping).
    assert (
        set(unique_merged_indices) == set(allowed_indices)
        if drop_long_sample
        else set(unique_merged_indices).issuperset(allowed_indices)
    )

    # Check that the total number of items across all splits is correct for the allowed items.
    num_total_items_in_result = sum(len(r) for r in result)
    if drop_long_sample:
        assert num_total_items_in_result == len(allowed_indices)
    else:
        # If not dropping, we also have singletons for large items
        num_big_items = len(length_col[~allowed_mask])
        expected_total = len(allowed_indices) + num_big_items
        assert num_total_items_in_result == expected_total


def test_compute_length_splits_edge_cases() -> None:
    """Test some edge cases to ensure function handles them properly."""
    # 1. Empty array
    length_col = np.array([], dtype=np.int32)
    splits = compute_length_splits(length_col, max_tokens=5)
    assert splits == []

    # 2. All items identical (no sorting difference)
    length_col = np.array([3, 3, 3, 3], dtype=np.int32)
    splits = compute_length_splits(length_col, max_tokens=6, order_by_length=False)
    # max_len=3, adding the second item => 3*2=6 => OK, adding the third => 3*3=9 => too big
    # So we expect splits of size 2, 2 if no sorting is applied
    assert len(splits) == 2
    assert all(len(s) == 2 for s in splits)

    # 3. Single item larger than max_tokens with drop_long_sample=True
    length_col = np.array([10], dtype=np.int32)
    splits = compute_length_splits(length_col, max_tokens=5, drop_long_sample=True)
    # Should be empty since 10 > 5
    assert len(splits) == 0

    # 4. Single item larger than max_tokens with drop_long_sample=False
    splits = compute_length_splits(length_col, max_tokens=5, drop_long_sample=False)
    # Should have 1 chunk containing that one large item
    assert len(splits) == 1
    assert len(splits[0]) == 1
    assert splits[0][0] == 0

    # 5. Check ordering effect explicitly
    # Let's set up an array [2, 8, 5], max_tokens=10
    # - If order_by_length=True => sorted => [2,5,8], chunking might differ from no sorting
    length_col = np.array([2, 8, 5], dtype=np.int32)

    splits_sorted = compute_length_splits(
        length_col, max_tokens=10, order_by_length=True
    )
    # Sorted => [2,5,8].
    #   Start chunk with 2 => can add 5 => new max_len=5 => 5*2=10 => OK => can we add 8?
    #   new max_len=8 => 8*3=24 => too big => finalize chunk [2,5], next chunk => [8]
    # So we expect 2 splits total.
    assert len(splits_sorted) == 2

    splits_unsorted = compute_length_splits(
        length_col, max_tokens=10, order_by_length=False
    )
    # Unsorted => [2,8,5].
    #   Start chunk with 2 => can we add 8 => max_len=8 => 8*2=16 => too big => finalize chunk [2]
    #   Next chunk => start at 8 => can we add 5 => max_len=8 => 8*2=16 => too big => finalize chunk [8]
    #   Next chunk => [5]
    # So we expect 3 splits total.
    assert len(splits_unsorted) == 3


class TestComputeRowsLength(unittest.TestCase):
    def test_list_array(self) -> None:
        arr = pa.array([[1, 2], [], [3]])
        lengths = compute_rows_length(arr)
        assert np.array_equal(lengths, np.array([2, 0, 1], dtype=np.int32))

    def test_empty_array(self) -> None:
        arr = pa.array([], type=pa.list_(pa.int64()))
        lengths = compute_rows_length(arr)
        assert len(lengths) == 0
