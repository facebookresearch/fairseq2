import typing as tp
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pyarrow.dataset import get_partition_keys  # requires pyarrow >= 13
from torch import Tensor

from fairseq2.data import CString


@contextmanager
def pyarrow_cpu(nb_cpu: int) -> tp.Generator[None, None, None]:
    nb_cpu_old = pa.cpu_count()
    nb_io_cpu_old = pa.io_thread_count()
    pa.set_cpu_count(nb_cpu)
    pa.set_io_thread_count(nb_cpu)
    try:
        yield
    finally:
        pa.set_cpu_count(nb_cpu_old)
        pa.set_io_thread_count(nb_io_cpu_old)


NestedDict = tp.Dict[str, "NestedDictValue"]
NestedDictValue = tp.Union[torch.Tensor, tp.List[CString], pd.Series, NestedDict]

BatchOutputType = tp.Union[pa.Table, pd.DataFrame, NestedDict]


def from_pyarrow_to_torch_tensor(
    arr: tp.Union[pa.Array, pa.ChunkedArray], strict: bool = True
) -> NestedDictValue:
    """
    struct_array = pa.Array.from_pandas([{"x": 4, "y": "RR"}] * 10)
    nest_array = pa.Array.from_pandas([[{'a': 1}, {'a': 2}]])
    """
    # for future ideas https://arrow.apache.org/docs/python/generated/pyarrow.Tensor.html
    # for sparse matrix support https://github.com/apache/arrow/blob/main/python/pyarrow/tests/test_sparse_tensor.py

    assert arr.null_count == 0, "does not support null values yet"

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


def batch_collater(
    inp: tp.List[Tensor], padding: tp.Optional[int]
) -> tp.Dict[str, tp.Union[bool, Tensor]]:
    # TODO: replace it with fairseq2 Collater
    seq_lens = torch.IntTensor([x.shape[0] for x in inp])
    return {
        "seqs": torch.nested.to_padded_tensor(
            torch.nested.as_nested_tensor(inp), padding=padding
        ),
        "seq_lens": seq_lens,
        # "is_ragged": False
        # if len(seq_lens) == 0
        # else bool((seq_lens != seq_lens[0]).any().item()),
    }


def map_structure(func, nested_object):  # type: ignore
    """Map a function over torch.Tensor in a (possibly nested) collection.
    Similar `to tf.nest.map_structure`.
    See also https://texar-pytorch.readthedocs.io/en/latest/_modules/texar/torch/utils/utils.html#map_structure
    """
    if isinstance(nested_object, list):
        return [map_structure(func, x) for x in nested_object]
    if isinstance(nested_object, tuple):
        if isinstance(nested_object, torch.Size):
            return func(nested_object)
        if hasattr(nested_object, "_fields"):  # namedtuple
            return type(nested_object)(*[map_structure(func, x) for x in nested_object])
        else:
            return tuple(map_structure(func, x) for x in nested_object)

    if isinstance(nested_object, dict):
        return {k: map_structure(func, v) for k, v in nested_object.items()}
    if isinstance(nested_object, set):
        return {map_structure(func, x) for x in nested_object}
    if isinstance(nested_object, torch.Tensor):
        return func(nested_object)
    else:
        return nested_object


def init_parquet_dataset(
    parquet_path: str,
    filters: tp.Optional[pa.dataset.Expression] = None,
    filesystem: tp.Optional[pa.fs.FileSystem] = None,
) -> pq.ParquetDataset:
    source_ds = pq.ParquetDataset(
        parquet_path,
        validate_schema=True,
        filters=filters,
        filesystem=filesystem,
    )
    return source_ds


def get_dataset_fragments(
    dataset: pq.ParquetDataset, filters: pa.dataset.Expression
) -> tp.List[pa.dataset.Fragment]:
    """
    This could be simplified once `split_row_groups=True` is implemented at `pq.ParquetDataset`.
    We could also return a generator instead of list (when getting full infos from S3 may be slow)
    """
    return list(dataset._dataset.get_fragments(filters))


def split_fragment_in_row_groups(
    fragment: pa.dataset.Fragment,
) -> tp.List[pa.dataset.Fragment]:
    return list(fragment.split_by_row_group())


def add_partitioning_values(
    table: pa.Table, fragment: pa.dataset.Fragment, columns: tp.Optional[tp.List[str]]
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
    fragment: pa.dataset.Fragment, columns: tp.Optional[tp.List[str]] = None
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
    filters: tp.Optional[pa.dataset.Expression] = None,
    drop_null: bool = True,
) -> pa.Table:
    if drop_null:
        table = table.drop_null()
    if filters is not None:
        table = table.filter(filters)
    return table


def concat_table(tables: tp.List[pa.Table]) -> pa.Table:
    return pa.concat_tables(
        tables,
        promote_options="permissive",  # needed to get deal with empty segments
    ).combine_chunks()


def compute_length_splits(
    length_col: npt.NDArray[np.int32], max_tokens: int
) -> tp.List[npt.NDArray[np.int32]]:
    """split sequence of length_col in the chunks such that total length is ~ max_tokens
        countint the padding to max length of elements in a chunk

    Args:
        length_col (np.ndarray):
        max_tokens (int):

    Returns:
        tp.List[np.ndarray]: splits that contain indices over the original length_col
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


def compute_rows_length(pa_array: pa.Array) -> npt.NDArray[np.int32]:
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
