import typing as tp
import torch
import pyarrow as pa
from torch import Tensor
import pandas as pd
from contextlib import contextmanager
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
