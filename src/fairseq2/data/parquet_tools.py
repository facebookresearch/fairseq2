# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from pyarrow.dataset import get_partition_keys  # requires pyarrow >= 13

from fairseq2.data import CString
from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder, read_sequence


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
def torch_random_seed(seed: Optional[int] = None) -> Generator[None, None, None]:
    if seed is not None:
        torch.manual_seed(seed)
    yield


NestedDict = Dict[str, "NestedDictValue"]
NestedDictValue = Union[torch.Tensor, List[CString], pd.Series, NestedDict]
BatchOutputType = Union[pa.Table, pd.DataFrame, NestedDict]


def from_pyarrow_to_torch_tensor(
    arr: Union[pa.Array, pa.ChunkedArray], strict: bool = True
) -> NestedDictValue:
    """
    struct_array = pa.Array.from_pandas([{"x": 4, "y": "RR"}] * 10)
    nest_array = pa.Array.from_pandas([[{'a': 1}, {'a': 2}]])
    """
    # for future ideas https://arrow.apache.org/docs/python/generated/pyarrow.Tensor.html
    # for sparse matrix support https://github.com/apache/arrow/blob/main/python/pyarrow/tests/test_sparse_tensor.py

    if arr.null_count != 0:
        raise ValueError("to torch conversion does not support null values")

    if isinstance(arr, pa.ChunkedArray):
        arr = arr.chunks[0] if arr.num_chunks == 1 else arr.combine_chunks()

    arr_type = arr.type
    if pa.types.is_primitive(arr_type):
        return torch.from_numpy(arr.to_numpy(zero_copy_only=True))

    try:
        return torch.from_numpy(arr.to_numpy(zero_copy_only=True))
    except pa.ArrowInvalid:
        pass

    if pa.types.is_dictionary(arr_type):
        return from_pyarrow_to_torch_tensor(arr.dictionary_decode())

    if pa.types.is_string(arr_type):
        return list(map(CString, arr.to_pandas()))

    if (
        pa.types.is_list(arr_type) or pa.types.is_large_list(arr_type)
    ) and pa.types.is_primitive(arr_type.value_type):
        return torch.nested.as_nested_tensor(
            list(map(torch.from_numpy, arr.to_pandas()))
        )

    if pa.types.is_fixed_size_list(arr_type) and pa.types.is_primitive(
        arr_type.value_type
    ):
        return torch.from_numpy(np.reshape(arr.values, (-1, arr_type.list_size)))

    if pa.types.is_struct(arr_type):
        return {
            arr_type.field(i).name: from_pyarrow_to_torch_tensor(arr.field(i))
            for i in range(arr_type.num_fields)
        }

    if pa.types.is_nested(arr_type):
        # TODO: deal with arr = [[{'a': 1}, {'a': 2}]]
        pass

    if strict:
        raise NotImplementedError(f"{arr_type} cannot be converted to torch.Tensor")
    else:
        return arr


def pyarrow_table_to_torch_dict(tt: pa.Table, strict: bool = True) -> NestedDict:
    return {
        col: from_pyarrow_to_torch_tensor(tt[col], strict) for col in tt.column_names
    }


def init_parquet_dataset(
    parquet_path: str,
    filters: Optional[pa.dataset.Expression] = None,
    filesystem: Optional[pa.fs.FileSystem] = None,
) -> pq.ParquetDataset:
    return pq.ParquetDataset(parquet_path, filters=filters, filesystem=filesystem)


def get_dataset_fragments(
    dataset: pq.ParquetDataset, filters: pa.dataset.Expression
) -> List[pa.dataset.Fragment]:
    """
    This could be simplified once `split_row_groups=True` is implemented at `pq.ParquetDataset`.
    We could also return a generator instead of list (when getting full infos from S3 may be slow)
    """
    return list(dataset._dataset.get_fragments(filters))


def split_fragment_in_row_groups(
    fragment: pa.dataset.Fragment,
) -> List[pa.dataset.Fragment]:
    return list(fragment.split_by_row_group())


def add_partitioning_values(
    table: pa.Table, fragment: pa.dataset.Fragment, columns: Optional[List[str]]
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
    fragment: pa.dataset.Fragment, columns: Optional[List[str]] = None
) -> pa.Table:
    fragment_columns = columns
    if fragment_columns is not None:
        fragment_columns = [
            col for col in fragment_columns if col in fragment.physical_schema.names
        ]
    fragment_table = fragment.to_table(columns=fragment_columns, use_threads=False)
    fragment_table = add_partitioning_values(fragment_table, fragment, columns)
    return fragment_table


def apply_filter(
    table: pa.Table,
    filters: Optional[pa.dataset.Expression] = None,
    drop_null: bool = True,
) -> pa.Table:
    if drop_null:
        table = table.drop_null()
    if filters is not None:
        table = table.filter(filters)
    return table


def concat_table(tables: List[pa.Table], combine: bool = True) -> pa.Table:
    result = pa.concat_tables(
        tables,
        promote_options="permissive",  # needed to get deal with empty segments
    )
    if combine:
        result = result.combine_chunks()
    return result


def compute_length_splits(
    length_col: NDArray[np.int32], max_tokens: int
) -> List[NDArray[np.int32]]:
    """split sequence of length_col in the chunks such that total length is ~ max_tokens
        countint the padding to max length of elements in a chunk

    Args:
        length_col (np.ndarray):
        max_tokens (int):

    Returns:
        List[np.ndarray]: splits that contain indices over the original length_col
    """
    argsort_ind = np.argsort(length_col)
    # TODO: remove 0 lengths
    sorted_length_col = length_col[argsort_ind]

    splits = []
    ptr = 0
    for i, length in enumerate(sorted_length_col):
        if length * (i - ptr) > max_tokens:
            splits.append(argsort_ind[ptr : (i - 1)])
            ptr = i - 1
    if (
        length <= max_tokens
    ):  # we drop the last iteration if it results in a batch greater than max_tokens
        splits.append(argsort_ind[ptr:])
    return splits


def compute_rows_length(pa_array: pa.Array) -> NDArray[np.int32]:
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


def _to_real_object(x: Union[_TableWrapper, NestedDict]) -> BatchOutputType:
    if isinstance(x, _TableWrapper):
        return x.table
    elif isinstance(x, list):
        return [_to_real_object(e) for e in x]
    elif isinstance(x, tuple):
        return tuple(_to_real_object(e) for e in x)
    else:
        return x


def table_func_wrap(func):  # type: ignore
    def inner(*args):  # type: ignore
        fixed_args = [_to_real_object(x) for x in args]
        result = func(*fixed_args)
        if isinstance(result, (pa.Table, pd.DataFrame)):
            result = _TableWrapper(result)
        return result

    return inner


def list_parquet_fragments(
    parquet_path: str,
    filters: Optional[pa.dataset.Expression] = None,
    columns: Optional[List[str]] = None,
    split_to_row_groups: bool = True,
    filesystem: Optional[pa.fs.FileSystem] = None,
    shuffle_window: Optional[int] = None,
    seed: Optional[int] = None,
) -> DataPipelineBuilder:
    dataset = init_parquet_dataset(parquet_path, filters=filters, filesystem=filesystem)
    columns = columns or dataset.schema.names
    if not set(columns).issubset(set(dataset.schema.names)):
        raise ValueError(
            f"columns {sorted(set(columns) - set(dataset.schema.names))} are not found in the dataset schema"
        )

    pipeline_builder = read_sequence(get_dataset_fragments(dataset, filters))

    with torch_random_seed(seed):
        if shuffle_window is not None:
            # shuffle them in full memory since fragments are already known
            pipeline_builder = pipeline_builder.shuffle(shuffle_window=0)

        if split_to_row_groups:
            pipeline_builder = pipeline_builder.yield_from(
                lambda fragment: read_sequence(
                    split_fragment_in_row_groups(fragment)
                ).and_return()
            )
            if shuffle_window is not None:
                pipeline_builder = pipeline_builder.shuffle(
                    shuffle_window=shuffle_window
                )

    return pipeline_builder


def build_iterator_over_one_table(
    table: pa.Table,
    order_by_length: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    num_parallel_calls: int = 8,
) -> DataPipeline:
    random_state = np.random.RandomState(seed)
    if order_by_length is not None:
        length_col = compute_rows_length(table[order_by_length])
        # add small perturbation to avoid same sample appear together during different epochs
        if shuffle:
            perturbation = random_state.randint(
                0,
                np.quantile(length_col, 0.001).astype(np.int32) + 2,
                len(length_col),
            )
            length_col += np.asarray(perturbation, dtype=np.int32)
    else:
        if shuffle:
            length_col = random_state.randint(0, 2**23, len(table))
        else:
            length_col = np.zeros(len(table), dtype=np.int32)

    if batch_size is not None:
        order_tt = pa.Table.from_arrays(
            [pa.array(np.argsort(length_col, kind="stable"))], ["order"]
        )
        batches = [ind["order"] for ind in order_tt.to_batches(batch_size)]
    elif max_tokens is not None:
        batches = compute_length_splits(length_col, max_tokens)
    else:
        raise ValueError("unknown batching method")

    if shuffle:
        batches = [batches[i] for i in random_state.permutation(len(batches))]

    return (
        read_sequence(batches)
        .map(
            table_func_wrap(lambda ind: table.take(ind).combine_chunks()),
            num_parallel_calls=num_parallel_calls,
        )
        .and_return(max_num_warnings=4)
    )
