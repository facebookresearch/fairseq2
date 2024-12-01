from typing import Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def is_list_like(arr):
    return pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)


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
