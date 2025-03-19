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
from typing import Any, Generator, List, Optional, Union, no_type_check

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


def simple_array_to_nested(arr: pa.ChunkedArray | pa.Array) -> pa.Array:
    """
    >>> a = pa.array([1,2,3])
    >>> simple_array_to_nested(a).to_pylist()
    [[1], [2], [3]]
    """
    return pa.ListArray.from_arrays(
        pa.array(np.arange(len(arr) + 1, dtype=np.int32)), pyarrow_column_to_array(arr)
    )


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


def add_fragments_trace(table: pa.Table, fragment: pa.dataset.Fragment) -> pa.Table:
    # we assume that the table will be loaded in the same order as the row groups
    row_group_ids = np.repeat(
        np.array([rr.id for rr in fragment.row_groups]),
        np.array([rr.num_rows for rr in fragment.row_groups]),
    )
    row_group_ids = row_group_ids.astype(np.int32)
    assert len(row_group_ids) == len(table)

    index_in_fragment = np.concatenate(
        [np.arange(rr.num_rows, dtype=np.int32) for rr in fragment.row_groups]
    )

    table = table.append_column(
        "__row_groups_ids",
        pa.array(row_group_ids, type=pa.int32()),
    )
    table = table.append_column(
        "__index_in_fragement", pa.array(index_in_fragment, type=pa.int32())
    )

    return table


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
