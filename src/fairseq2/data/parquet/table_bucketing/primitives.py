from functools import reduce
from typing import List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from fairseq2.data.data_pipeline import DataPipeline, read_sequence
from fairseq2.data.parquet.arrow_transform import is_list_like
from fairseq2.logging import log


def compute_length_splits(
    length_col: Union[NDArray[np.int32], pa.Array],
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


def compute_rows_length(
    pa_array: Union[pa.Array, pa.ChunkedArray],
) -> NDArray[np.int32]:
    """
    Compute the length of each row in a PyArrow array.

    This function handles the following types:
        - Integers/Floats: Uses as it !
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
    if pa.types.is_integer(type_) or pa.types.is_floating(type_):
        length_col = pa_array.to_numpy(zero_copy_only=False).copy()
    elif is_list_like(pa_array):
        length_col = (
            pc.list_value_length(pa_array).to_numpy(zero_copy_only=False).copy()
        )
    elif pa.types.is_string(type_):
        # TODO: use polars to compute chat length
        length_col = pc.utf8_length(pa_array).to_numpy().copy(zero_copy_only=False)
    else:
        length_col = np.asarray(pa_array.to_pandas().apply(len), dtype=np.int32)

    length_col[np.isnan(length_col)] = 0
    return length_col


def build_batching_loop_over_one_table(
    table: pa.Table,
    order_by_length: bool = False,
    length_columns: Optional[List[Optional[str]]] = None,
    batch_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    drop_long_sample: bool = True,
    shuffle: bool = False,
    len_reducer: Optional[str] = None,
    seed: Optional[int] = None,
    num_parallel_calls: int = 1,
) -> DataPipeline:

    if max_tokens is None and batch_size is None:
        raise ValueError("Need to provide either max_tokens or batch_size")

    if (max_tokens is not None) or order_by_length:
        assert length_columns is not None, "Need to provide length columns"

    if batch_size is not None and not order_by_length and not shuffle:
        # early exit if we don't need to shuffle, (.slice instead of .take)
        return read_sequence(
            [
                pa.Table.from_batches([x], schema=table.schema)
                for x in table.to_batches(max_chunksize=batch_size)
            ]
        ).and_return()

    random_state = np.random.RandomState(seed)

    multiple_length_reducer = {"max": np.maximum, "min": np.minimum, "sum": np.add}

    if length_columns is not None and len(length_columns) > 0:
        length_col = reduce(
            multiple_length_reducer[len_reducer or "sum"],
            (compute_rows_length(table[lc]) for lc in length_columns),
        )
    else:
        if shuffle:
            length_col = random_state.randint(0, 2**23, len(table))
        else:
            length_col = np.zeros(len(table), dtype=np.int32)
            # we should not be there...

    if batch_size is not None:
        if order_by_length:
            sorting_ind = np.argsort(length_col, kind="stable")
        else:
            sorting_ind = np.arange(len(length_col), dtype=np.int32)

        order_tt = pa.Table.from_arrays([pa.array(sorting_ind)], ["order"])
        batches = [ind["order"] for ind in order_tt.to_batches(batch_size)]
    elif max_tokens is not None:
        batches = compute_length_splits(
            length_col,
            max_tokens,
            order_by_length=order_by_length,
            drop_long_sample=drop_long_sample,
        )
    else:
        raise ValueError("unknown batching method")

    if shuffle:
        batches = [batches[i] for i in random_state.permutation(len(batches))]

    def _getter(ind):
        try:
            return table.take(ind)
        except Exception as e:
            log.warn(f"Unexpected error : \n {str(e)} \n {table} \n {ind}")
            return None

    return (
        read_sequence(batches)
        .map(_getter, num_parallel_calls=num_parallel_calls)
        .filter(lambda tt: bool(tt is not None))
        .and_return(max_num_warnings=4)
    )
