# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Union, no_type_check

import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from pyarrow.dataset import get_partition_keys  # requires pyarrow >= 13

from tqdm.auto import tqdm

from fairseq2.data import DataPipeline, DataPipelineBuilder, read_sequence
from fairseq2.data.parquet.arrow import pyarrow_column_to_array
from fairseq2.logging import get_log_writer

logger = get_log_writer(__name__)


@contextmanager
def pyarrow_cpu(nb_cpu: int) -> Generator[None, None, None]:
    nb_cpu_old = pa.cpu_count()
    nb_io_cpu_old = pa.io_thread_count()
    pa.set_cpu_count(nb_cpu)
    pa.set_io_thread_count(nb_cpu)
    try:
        yield
    finally:
        pa.set_cpu_count(nb_cpu_old)
        pa.set_io_thread_count(nb_io_cpu_old)


@contextmanager
def torch_random_seed(seed: int | None = None) -> Generator[None, None, None]:
    if seed is not None:
        torch.manual_seed(seed)
    yield

from fairseq2.logging import log

NestedDict = dict[str, "NestedDictValue"]
NestedDictValue = torch.Tensor | list[str] | pd.Series | NestedDict
BatchOutputType = pa.Table | pd.DataFrame | NestedDict


def is_list_like(arr: pa.Array) -> bool:
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
    if arr.offset == 0:
        return arr

    new_values = _fix_list_offset(pc.list_flatten(arr))
    new_offsets = pc.subtract(arr.offsets, arr.offsets[0])

    return (
        pa.LargeListArray.from_arrays(new_offsets, new_values)
        if pa.types.is_large_list(arr.type)
        else pa.ListArray.from_arrays(new_offsets, new_values)
    )


def pyarrow_column_to_array(arg: Union[pa.ChunkedArray, pa.Array]) -> pa.Array:
    # see https://github.com/apache/arrow/issues/37318
    if isinstance(arg, pa.Array):
        return _fix_list_offset(arg)

    return _fix_list_offset(
        arg.chunk(0) if arg.num_chunks == 1 else arg.combine_chunks()
    )


def hstack_pyarray_list(*arrays: Union[pa.ChunkedArray, pa.Array]) -> pa.Array:
    """
    Example with simple list:
    >>> a = pa.array([[1], [2,3], [5], []])
    >>> b = pa.array([[-1, -3], [-11], [], [22]])
    >>> hstack_pyarray_list(a, b).to_pylist()
    [[1, -1, -3], [2, 3, -11], [5], [22]]

    Example with nested lists:
    >>> data = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10]])]
    >>> list_array = nested_numpy_to_pyarrow(data)
    >>> list_array.type
    ListType(list<item: fixed_size_list<item: int64>[2]>)
    >>> truncated_list_array = pc.list_slice(list_array, 1, 2)
    [[[3, 4]], [[7, 8]], []]
    >>> hstack_pyarray_list(list_array, truncated_list_array)
    [[[1, 2], [3, 4], [3, 4]],
     [[5, 6], [7, 8], [7, 8]],
     [[9, 10]]]
    """
    if not all(map(is_list_like, arrays)):
        raise ValueError("All pyarrow arrays must be list-like")

    lens = list(set(map(len, arrays)))
    if len(lens) != 1:
        raise ValueError("All pyarrow arrays must have the same length")

    list_off_views = [
        pyarrow_column_to_array(pc.list_flatten(arr.slice(i, 1)))
        for i in range(lens[0])
        for arr in arrays
    ]

    is_large = any(pa.types.is_large_list(arr.type) for arr in arrays)

    offsets = np.concatenate(
        [
            np.array([0]),
            np.sum([pc.list_value_length(arr) for arr in arrays], axis=0),
        ],
        dtype=np.int64 if is_large else np.int32,
    ).cumsum()

    cls = pa.LargeListArray if is_large else pa.ListArray
    return cls.from_arrays(offsets, pa.concat_arrays(list_off_views))


@no_type_check
def pyarrow_to_torch_tensor(
    arr: Union[pa.Array, pa.ChunkedArray], strict: bool = False
) -> NestedDictValue:
    """
    struct_array = pa.Array.from_pandas([{"x": 4, "y": "RR"}] * 10)
    nest_array = pa.Array.from_pandas([[{'a': 1}, {'a': 2}]])
    """
    # for future ideas https://arrow.apache.org/docs/python/generated/pyarrow.Tensor.html
    # for sparse matrix support https://github.com/apache/arrow/blob/main/python/pyarrow/tests/test_sparse_tensor.py

    if arr.null_count != 0:
        raise ValueError("to torch conversion does not support null values")

    arr = pyarrow_column_to_array(arr)

    arr_type = arr.type
    if pa.types.is_primitive(arr_type):
        try:
            return torch.from_numpy(arr.to_numpy(zero_copy_only=True))
        except Exception:
            pass

    try:
        return torch.from_numpy(arr.to_numpy(zero_copy_only=True))
    except pa.ArrowInvalid:
        pass

    if pa.types.is_dictionary(arr_type):
        return pyarrow_to_torch_tensor(arr.dictionary_decode())

    if pa.types.is_string(arr_type):
        return arr.to_pandas().tolist()

    if pa.types.is_list(arr_type) or pa.types.is_large_list(arr_type):
        if pa.types.is_primitive(arr_type.value_type):
            return arr.to_pandas().map(torch.from_numpy).tolist()

        if pa.types.is_fixed_size_list(arr_type.value_type) and pa.types.is_primitive(
            arr_type.value_type.value_type
        ):
            # FIXME: get the column global dtype for empty seq case
            return (
                arr.to_pandas()
                .map(
                    lambda x: torch.from_numpy(
                        np.vstack(x) if len(x) > 0 else np.array([], dtype=np.float32)
                    )
                )
                .tolist()
            )

    if pa.types.is_fixed_size_list(arr_type):
        if pa.types.is_primitive(arr_type.value_type):
            return torch.from_numpy(np.reshape(arr.values, (-1, arr_type.list_size)))

    if pa.types.is_struct(arr_type):
        return {
            arr_type.field(i).name: pyarrow_to_torch_tensor(arr.field(i))
            for i in range(arr_type.num_fields)
        }

    if pa.types.is_nested(arr_type):
        raise NotImplementedError("Nested list is not supported")

    if strict:
        raise NotImplementedError(f"{arr_type} cannot be converted to torch.Tensor")
    else:
        return arr  # keeping as in the orignal pyarrow form


def pyarrow_table_to_torch_dict(tt: pa.Table, strict: bool = False) -> NestedDict:
    out = {}
    for col in tt.column_names:
        try:
            out[col] = pyarrow_to_torch_tensor(tt[col], strict)
        except ValueError as e:
            log.info(
                f"Column {col} of type {tt[col].type} was not converted to torch as expected",
                str(e),
            )
            out[col] = tt[col]
    return out


# --- tested above --- #


@contextmanager
def pyarrow_cpu(nb_cpu: int) -> Generator[None, None, None]:
    """
    Set the number of CPU cores to use for pyarrow.
    """
    nb_cpu_old = pa.cpu_count()
    nb_io_cpu_old = pa.io_thread_count()
    pa.set_cpu_count(nb_cpu)
    pa.set_io_thread_count(nb_cpu)
    try:
        yield
    finally:
        pa.set_cpu_count(nb_cpu_old)
        pa.set_io_thread_count(nb_io_cpu_old)


@contextmanager
def torch_random_seed(seed: Optional[int] = None) -> Generator[None, None, None]:
    """
    Set the random seed for torch.

    Args:
        seed (Optional[int]): The random seed to set. If None, the seed is not set.
    """
    if seed is not None:
        torch.manual_seed(seed)
    yield
>>>>>>> dd85d23b (update parquet utils)


def get_dataset_fragments(
    dataset: pq.ParquetDataset, filters: pa.dataset.Expression
) -> list[pa.dataset.Fragment]:
    """
    This could be simplified once `split_row_groups=True` is implemented at `pq.ParquetDataset`.
    We could also return a generator instead of list (when getting full infos from S3 may be slow)
    """
    return list(dataset._dataset.get_fragments(filters))


def split_fragment_in_row_groups(
    fragment: pa.dataset.Fragment,

) -> list[pa.dataset.Fragment]:
    """
    Split a fragment into multiple fragments by row groups.
    """
    return list(fragment.split_by_row_group())


def add_partitioning_values(
    table: pa.Table, fragment: pa.dataset.Fragment, columns: list[str] | None
) -> pa.Table:
    """
    When loading a single fragment, pyarrow does not add the partitioning columns,
    so we need to do it manually.
    """
    for key, val in get_partition_keys(fragment.partition_expression).items():
        if columns is None or key in columns:
            values = pa.DictionaryArray.from_arrays(
                np.zeros(len(table), dtype=np.int32), [val]
            )
            table = table.append_column(key, values)
    return table


def load_one_fragment(
    fragment: pa.dataset.Fragment, columns: list[str] | None = None
) -> pa.Table:
    """
    Load a single fragment from a dataset as a PyArrow Table.

    This function filters out any column names not present in the fragment's
    physical schema, then loads the data from the fragment as a PyArrow Table.
    Finally, it calls `add_partitioning_values` to include partition keys
    (if that function is defined to do so).

    Args:
        fragment (pa.dataset.Fragment): The dataset fragment to load.
        columns (Optional[List[str]]): An optional list of column names to load.
            If None, all columns in the fragment's physical schema are used.

    Returns:
        pa.Table: A PyArrow Table containing data from the fragment (and
        potentially augmented with partitioning values).

    Raises:
        ValueError: If the fragment is invalid or cannot be read. (You can
            optionally raise or handle other exceptions as needed.)
    """
    # Ensure columns only include names that exist in the fragment
    fragment_columns = columns
    if fragment_columns is not None:
        fragment_columns = [
            col for col in fragment_columns if col in fragment.physical_schema.names
        ]
    # Load the fragment data into a PyArrow table
    fragment_table = fragment.to_table(columns=fragment_columns, use_threads=False)

    # Add partitioning values if the function is defined
    fragment_table = add_partitioning_values(fragment_table, fragment, columns)

    return fragment_table


def compute_length_splits(
    length_col: NDArray[np.int32],
    max_tokens: int,
    *,
    order_by_length: bool = True,
    drop_long_sample: bool = True,
) -> List[NDArray[np.int32]]:
    """
    Split a sequence of lengths (`length_col`) into chunks so that
    the total "padded" length in each chunk is ~ `max_tokens`.
    The "padded" length is computed as the max length in the chunk
    multiplied by the number of items in that chunk.

    Args:
        length_col (np.ndarray): Array of sequence lengths.
        max_tokens (int): Maximum tokens allowed in a chunk
                          based on padding to the max length in the chunk.
        order_by_length (bool): If True, sort the sequences by length before splitting.
        drop_long_sample (bool): If True, drop any items whose length exceeds `max_tokens`.

    Returns:
        list[np.ndarray]: A list of arrays, each containing the indices of
                          the original `length_col` that belong to that split.
    """
    argsort_ind = (
        np.argsort(length_col)
        if order_by_length
        else np.arange(len(length_col), dtype=np.int32)
    )

    sorted_length_col = length_col[argsort_ind]

    small_elements_masks = sorted_length_col <= max_tokens
    big_elements_inds = argsort_ind[~small_elements_masks]

    argsort_ind = argsort_ind[small_elements_masks]
    sorted_length_col = sorted_length_col[small_elements_masks]

    size = len(sorted_length_col)
    splits = []
    begin, end = 0, 0
    while end < size:
        current_max_len = sorted_length_col[begin]
        begin = end
        while end < size:
            current_max_len = max(current_max_len, sorted_length_col[end])
            if current_max_len * (end + 1 - begin) > max_tokens:
                splits.append(argsort_ind[begin:end])
                break
            end += 1
    else:
        if begin < size:
            splits.append(argsort_ind[begin:])

    # adding big sample at the end one by one
    if not drop_long_sample and len(big_elements_inds):
        splits.extend(np.array_split(big_elements_inds, len(big_elements_inds)))

    return splits


def compute_rows_length(pa_array: pa.Array) -> NDArray[np.int32]:
    """
    Compute the length of each row in a PyArrow array.

    This function handles the following types:
        - List / LargeList: Uses pyarrow.compute.list_value_length
        - String: Uses pyarrow.compute.utf8_length
        - Fallback: Tries to convert to pandas and apply len (e.g., for arrays of Python objects)

    Null values (NaNs) are set to 0 in the returned array.

    Args:
        pa_array (pa.Array): PyArrow array whose element lengths should be computed.

    Returns:
        NDArray[np.int32]: NumPy array of the computed lengths for each element.
    """
    type_ = pa_array.type
    if pa.types.is_list(type_) or pa.types.is_large_list(type_):
        length_col = pa.compute.list_value_length(pa_array).to_numpy()
    elif pa.types.is_string(type_):
        length_col = pa.compute.utf8_length(pa_array).to_numpy()
    else:
        length_col = np.asarray(pa_array.to_pandas().apply(len))

    length_col = length_col.copy()
    length_col[np.isnan(length_col)] = 0
    return np.asarray(length_col, dtype=np.int32)


class _TableWrapper:
    """
    class to avoid fairseq2 casting pa.Table to iterable objects
    which currently fails
    """

    def __init__(self, table: pa.Table) -> None:
        self.table: pa.Table = table


def _to_real_object(x: _TableWrapper | NestedDict) -> BatchOutputType:
    if isinstance(x, _TableWrapper):
        return x.table
    elif isinstance(x, list):
        return [_to_real_object(e) for e in x]
    elif isinstance(x, tuple):
        return tuple(_to_real_object(e) for e in x)
    else:
        return x


def table_func_wrap(func: Callable[..., Any]) -> Callable[..., Any]:
    def inner(*args: Any) -> Any:
        fixed_args = [_to_real_object(x) for x in args]
        result = func(*fixed_args)
        if isinstance(result, (pa.Table, pd.DataFrame)):
            result = _TableWrapper(result)
        return result

    return inner
