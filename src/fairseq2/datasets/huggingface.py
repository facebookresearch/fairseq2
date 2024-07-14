# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, Callable, Dict, Optional


from fairseq2.data.data_pipeline import (
    Collater,
    DataPipelineBuilder,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.datasets.data_reader import DataPipelineReader
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch

try:
    from datasets import Dataset, DatasetDict  # type: ignore[attr-defined,import-untyped]
except ImportError:
    has_datasets = False
else:
    has_datasets = True


Example = Dict[str, Any]


class Batcher(abc.ABC):
    @abc.abstractmethod
    def batch(self, builder: DataPipelineBuilder, **kwargs: Any) -> DataPipelineBuilder:
        ...


class BatcherBySize(Batcher):
    def __init__(self, bucket_size: int) -> None:
        if bucket_size <= 0:
            raise ValueError(f"Forbidden value: bucket_size={bucket_size}")
        self.bucket_size = bucket_size

    def batch(self, builder: DataPipelineBuilder, **kwargs: Any) -> DataPipelineBuilder:
        return builder.bucket(bucket_size=self.bucket_size)


class BatcherBySeqLength(Batcher):
    """
    The batching strategy that looks into the sequence defined in the column
    `seq_len_col`, and batch them until certain accumulated length is reached.

    :param max_num_elements:
        The maximum number of elements in each batch.
    :param max_seq_len:
        The maximum sequence length of each example.
        Examples longer than this value will be cropped.
    :param min_seq_len:
        The minimum sequence length of each example. Examples
        shorter than this value will be dropped.
    :param seq_len_col:
        The column that stores length of source sequence. Must
        be provided if the data is to be batched by length
    """

    def __init__(
        self,
        max_num_elements: int,
        max_seq_len: Optional[int] = None,
        min_seq_len: Optional[int] = None,
        seq_len_col: Optional[str] = None,
        **extra: Any,
    ):
        self.max_num_elements = max_num_elements
        self.max_seq_len = max_seq_len or max_num_elements
        self.min_seq_len = min_seq_len or 1
        self.skip_below_min_examples = min_seq_len is not None
        self.seq_len_col = seq_len_col

    def batch(
        self, builder: DataPipelineBuilder, num_seqs_multiple_of: int = 1, **kwargs: Any
    ) -> DataPipelineBuilder:
        bucket_sizes = create_bucket_sizes(
            max_num_elements=self.max_num_elements,
            max_seq_len=self.max_seq_len,
            min_seq_len=self.min_seq_len,
            num_seqs_multiple_of=num_seqs_multiple_of,
        )

        return builder.bucket_by_length(
            bucket_sizes,
            selector=self.seq_len_col,
            min_data_len=self.min_seq_len,
            skip_below_min_examples=self.skip_below_min_examples,
            skip_above_max_examples=True,
        )


def create_hf_reader(
    dataset: Dataset,
    gang: Gang,
    converter: Callable[[Example], Seq2SeqBatch],
    *,
    batcher: Optional[Batcher] = None,
    num_accumulate: int = 1,
    num_prefetch: int = 1,
    **extra: Any,
) -> DataPipelineReader[Seq2SeqBatch]:
    """
    Convert HF Dataset into a fairseq2 PipelineReader.
    The HF-specific processing logic is assumed to be
    done before the conversion.

    **Batching**: The batching can be done externally or using
    bucketing strategies in fairseq2. If both ``bucket_size``
    and ``max_num_elements`` are None, then we assume the
    ``dataset`` is already batched and do nothing. Otherwise:

    - If ``bucket_size`` is not None, generate batches of fixed size
    - If ``max_num_elements`` is not None, generate batches up to
    `max_num_elements` size. In this case, ``max_seq_len`` must be
    given.

    In addition, the data
    should be prepared to have a column that contains
    sequence length for each example.

    This function should not be used for big datatset.

    :param gang:
        The gang over which to shard the dataset.
    :param batcher:
        The batching strategy that is used to transform the pipeline
        into iteration of batch of items.
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

    if batcher is None:
        data = dataset.data.to_batches()  # list of RecordBatch
    else:
        data = dataset.to_list()
    builder = read_sequence(data)

    # Shard.
    builder.shard(gang.rank, gang.size, allow_uneven=True)

    if batcher is None:
        # Convert RecordBatch to python dictionary
        builder = builder.map(lambda batch: batch.to_pydict())
    else:
        builder = batcher.batch(builder, **extra)

        # collate to python dict
        builder.map(Collater())

    # Prefetch `num_prefetch` examples in background.
    builder.prefetch(num_prefetch)

    pipeline = builder.map(converter).and_return()
    return DataPipelineReader[Seq2SeqBatch](
        pipeline,
        gang,
        num_accumulate=num_accumulate,
        drop_remainder=False,
        sync_batches=True,
    )
