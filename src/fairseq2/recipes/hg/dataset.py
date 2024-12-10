# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fairseq2.data.data_pipeline import Collater, create_bucket_sizes, read_sequence
from fairseq2.datasets.batching import Batching, LengthBatching, StaticBatching
from fairseq2.datasets.data_reader import BatchT, DataPipelineReader
from fairseq2.gang import Gang

try:
    from datasets import (  # type: ignore[attr-defined,import-untyped,import-not-found]
        Dataset,
        DatasetDict,
    )
except ImportError:
    has_datasets = False
else:
    has_datasets = True

Example = dict[str, Any]


def create_hf_reader(
    dataset: "Dataset",
    gang: Gang,
    converter: Callable[[Example], BatchT],
    *,
    batching: Batching | None = None,
    max_seq_len: int | None = None,
    drop_remainder: bool = False,
    min_seq_len: int = 0,
    seq_len_col: str | None = None,
    num_accumulate: int = 1,
    num_prefetch: int = 1,
    pad_value: int | None = None,
    **extra: Any,
) -> DataPipelineReader[BatchT]:
    """
    Convert HF Dataset into a fairseq2 PipelineReader.
    The HF-specific processing logic is assumed to be
    done before the conversion.

    **NOTE**: This function should not be used for big datatset, but
    as a convenient wrapper over some small HuggingFace datasets for
    quick iteration.

    **Batching**:

    :param gang:
        The gang over which to shard the dataset.
    :param batching:
        The batching strategy that is used to transform the pipeline
        into iteration of batch of items. If None, The batching is
        done externally using the datasets API.
    :param drop_remainder:
        Whether or not drop the remainder items that do not fit the
        batching strategy (e.g. not having the batch size)
    :param max_seq_len:
        used when batching by length, to define the cap of the batch
    :param min_seq_len:
        used to filter the items with too short length
    :param seq_len_col:
        The column storing example's length
    :param num_accumulate:
        The number of batches to accumulate in each iteration.
        Typically used with gradient accumulation during
        training.
    :param num_prefetch:
        The number of batches to prefetch in background.
    :param extras:
        The extra parameters specific to the dataset
        implementation.
    """
    if not has_datasets:
        raise ModuleNotFoundError(
            "`datasets` is required but not found. Please install it with `pip install datasets`."
        )  # fmt: skip

    # Make sure the dataset is a proper arrow dataset
    if not isinstance(dataset, Dataset):
        # One common mistake is pass a DatasetDict (e.g. with all splits) as inputs
        if isinstance(dataset, DatasetDict):
            raise TypeError(
                "create_reader() expects input of datasets.Dataset type, "
                "get datasets.DatasetDict. Make sure you specify `split` "
                "when loading the dataset"
            )
        raise TypeError(f"Expect datasets.Dataset type, get {type(dataset)}")

    if batching is None:
        data = dataset.data.to_batches()  # list of RecordBatch
    else:
        data = dataset
    builder = read_sequence(data)

    # Shard.
    builder.shard(gang.rank, gang.size, allow_uneven=True)

    if batching:
        if max_seq_len is None:
            raise ValueError(
                "`max_seq_len` is required if batching strategy is specified"
            )

        if isinstance(batching, LengthBatching):
            bucket_sizes = create_bucket_sizes(
                max_seq_len=max_seq_len,
                min_seq_len=min_seq_len,
                max_num_elements=batching.max_num_elements,
            )

            # Bucket by the length of the source or target sequence. The longer
            # one will be considered the length of the example.
            builder.bucket_by_length(
                bucket_sizes,
                selector=seq_len_col,
                min_data_len=min_seq_len,
                skip_below_min_examples=True,
                skip_above_max_examples=True,
                drop_remainder=drop_remainder,
            )
        elif isinstance(batching, StaticBatching):
            if seq_len_col:

                def skip(example: Example) -> bool:
                    _len = len(example[seq_len_col])
                    return _len >= min_seq_len and _len <= max_seq_len

                builder.filter(skip)

            builder = builder.bucket(batching.batch_size, drop_remainder=drop_remainder)
        else:
            raise RuntimeError(f"`{batching}` is not supported.")

        # collate to python dict
        builder.map(Collater(pad_value=pad_value))
    else:
        # Convert RecordBatch to python dictionary
        builder = builder.map(lambda batch: batch.to_pydict())

    # Prefetch `num_prefetch` examples in background.
    builder.prefetch(num_prefetch)

    pipeline = builder.map(converter).and_return()
    return DataPipelineReader[BatchT](
        pipeline,
        gang,
        num_accumulate=num_accumulate,
        drop_remainder=False,
        sync_batches=True,
    )
