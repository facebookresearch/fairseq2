# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import uuid
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Union, no_type_check

import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
import pyarrow.parquet as pq
import torch
import xxhash
from pyarrow.dataset import get_partition_keys  # requires pyarrow >= 13

from fairseq2.logging import log

NestedDict = dict[str, "NestedDictValue"]
NestedDictValue = torch.Tensor | list[str] | pd.Series | NestedDict
BatchOutputType = pa.Table | pd.DataFrame | NestedDict


def return_none_on_failure(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log.warning(f"An error occurred: {e}")
            return None

    return wrapper


def circular_shift_left(lst: List[Any], k: int) -> List[Any]:
    if len(lst) <= 1:
        return lst

    k = k % len(lst)  # To handle shifts larger than the list length
    return lst[k:] + lst[:k]


def fragment_stable_hash(
    fragment: pa.dataset.Fragment, seed: Optional[int] = None
) -> int:
    serialized_info = f"{fragment.path}-{[rr.id for rr in fragment.row_groups]}-{seed}"
    return xxhash.xxh32_intdigest(serialized_info)


def is_list_like(arr: Union[pa.ChunkedArray, pa.Array]) -> bool:
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


def pyarrow_column_to_array(arg: pa.ChunkedArray | pa.Array) -> pa.Array:
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


def get_dataset_fragments(
    dataset: pq.ParquetDataset, filters: pa.dataset.Expression
) -> list[pa.dataset.Fragment]:
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


def rename_table_columns(table: pa.Table, mapper: dict) -> pa.Table:
    output = table.rename_columns([mapper.get(key, key) for key in table.column_names])
    return output


def remove_file(file_name):
    try:
        os.remove(file_name)
    except FileNotFoundError:
        pass


def read_mmap_table_with_finalizer(file_name):
    with pa.memory_map(file_name, "rb") as source:
        table = pa.ipc.open_stream(source).read_all()
    weakref.finalize(table, remove_file, file_name)
    # XXX: this reference capture will not work properly in multiprocessing context
    return table


def write_table_to_arrow(table, cache_dir: str):
    current_name = f"table_{str(uuid.uuid4())[:15]}.arrow"
    file_path = str(Path(cache_dir).joinpath(current_name))
    with pa.OSFile(file_path, "wb") as current_sink:
        with pa.ipc.new_stream(current_sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=None)

    return file_path


def table_to_mmap_table(table, cache_dir: str):
    file_name = write_table_to_arrow(table, cache_dir)
    return read_mmap_table_with_finalizer(file_name)


def apply_on_nested_array(
    fn: Callable[[pa.Array], pa.Array], *inps: pa.ChunkedArray | pa.Array
) -> pa.ChunkedArray | pa.Array:

    if len(inps) > 0 and is_list_like(inps[0]):
        assert all(is_list_like(arr) for arr in inps)

        arrs = [pyarrow_column_to_array(arr) for arr in inps]
        assert all(arr.offset == 0 for arr in arrs)

        offsets = arrs[0].offsets
        cls = (
            pa.LargeListArray if pa.types.is_large_list(arrs[0].type) else pa.ListArray
        )

        assert all(pc.all(pc.equal(arr.offsets, offsets)).as_py() for arr in arrs)

        res = apply_on_nested_array(fn, *(pc.list_flatten(arr) for arr in arrs))
        output = cls.from_arrays(offsets=offsets, values=res)
        for arr in arrs:
            if arr.null_count > 0:
                output = pc.if_else(pc.is_null(arr), None, output)
        return output

    return fn(*inps)


def apply_over_groups(
    table: pa.Table,
    grp_columns: Optional[List[Optional[str]]],
    table_mapper: Callable[[pa.Table], pa.Table],
) -> pa.Table:
    """
    Apply a mapping function to each group of a PyArrow table.

    Parameters:
    - table: The input PyArrow table to be grouped and mapped.
    - grp_columns: A list of column names to group the table by.
        if `grp_columns=[None]` or `grp_columns=[]` or `grp_columns=None`,
        `table_mapper` is applied on the full table.
        Note also that None values in `grp_columns` will be filtered,
        so one can you use grp_columns=[col1, col2] where each of col1 and col2 can be None.
    - table_mapper: A callable function that takes a PyArrow table as input and returns a new PyArrow table.

    Returns:
    - A new PyArrow table resulting from applying the `table_mapper` function to each group of the input table.

    Notes:
    - The function adds a temporary column "__uuu_index" to the input table to facilitate grouping and sorting.
    - The `table_mapper` function is applied to each group of the table, and the resulting tables are concatenated,
        Therefore, the resulting sub-tables should have the same schema
    - The function adds a temporary column "__uuu_index" to keep track of the original order of the rows.
        So, it should be kept inchanged by `table_mapper`.
        This column is removed in the final result.
    """

    # shortcut for no group case
    if grp_columns is None:
        return table_mapper(table)

    grp_columns = [x for x in grp_columns if x is not None]
    if len(grp_columns) == 0:
        return table_mapper(table)

    table = table.append_column(
        "__uuu_index", pa.array(np.arange(len(table), dtype=np.int32))
    )
    split_grps = (
        table.select(grp_columns + ["__uuu_index"])
        .group_by(grp_columns)
        .aggregate([("__uuu_index", "list")])
    )
    # shortcut for single group case
    if len(split_grps) == 1:
        return table_mapper(table.drop("__uuu_index"))

    # to iterate per rows we convert to pandas
    # TODO : this could be called in parallel
    results = [
        table_mapper(table.take(pa.array(ind)))
        for ind in split_grps["__uuu_index_list"].to_pandas()
    ]

    result = pa.concat_tables(results, promote_options="permissive")
    del results

    if "__uuu_index" in result.column_names:
        return result.sort_by("__uuu_index").drop("__uuu_index")

    return result.combine_chunks()
