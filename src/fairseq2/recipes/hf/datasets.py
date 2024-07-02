# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Union, final

from datasets import Dataset  # type: ignore

from fairseq2.assets.card import AssetCard
from fairseq2.data.data_pipeline import Collater, create_bucket_sizes, read_sequence
from fairseq2.datasets.data_reader import DataPipelineReader
from fairseq2.datasets.loader import DatasetLoader
from fairseq2.gang import Gang

HFBatch = Dict[str, Any]


class HFDataset:
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self) -> str:
        return self.name

    def create_reader(
        self,
        dataset: Dataset,
        gang: Gang,
        bucket_size: Optional[int] = None,
        max_num_elements: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        min_seq_len: Optional[int] = None,
        seq_len_col: Optional[str] = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extra: Any,
    ) -> DataPipelineReader[HFBatch]:
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
        :param num_accumulate:
            The number of batches to accumulate in each iteration.
            Typically used with gradient accumulation during
            training.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param seed:
            The seed to initialize the random number generators
            used internally.
        :param extras:
            The extra parameters specific to the dataset
            implementation.
        """

        # Resolve the batching strategies
        def resolve_batching_strategy() -> int:
            if bucket_size is None and max_num_elements is None:
                return 0  # NO_BATCH
            elif bucket_size is not None and max_num_elements is not None:
                raise ValueError("Can only batch by size or by length, not both")
            elif bucket_size is not None:
                return 1  # BATCH_BY_SIZE
            else:
                assert (
                    max_seq_len
                ), "Max_seq_len must be specified in batch-by-length mode"
                return 2  # BATCH_BY_LEN

        batch_mode = resolve_batching_strategy()
        if batch_mode == 0:
            data = dataset.data.to_batches()  # list of RecordBatch
        else:
            data = dataset.to_list()
        builder = read_sequence(data)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        if batch_mode == 0:
            # Convert RecordBatch to python dictionary
            builder.map(lambda batch: batch.to_pydict())

        else:
            if batch_mode == 1:
                assert bucket_size
                builder.bucket(bucket_size=bucket_size)

            if batch_mode == 2:

                skip_below_min_examples = min_seq_len is not None
                min_seq_len = min_seq_len or 1

                # Bucket by audio length.
                assert max_num_elements and max_seq_len
                bucket_sizes = create_bucket_sizes(
                    max_num_elements=max_num_elements,
                    max_seq_len=max_seq_len,
                    min_seq_len=min_seq_len,
                    num_seqs_multiple_of=8,
                )

                builder.bucket_by_length(
                    bucket_sizes,
                    selector=seq_len_col,
                    min_data_len=min_seq_len,
                    skip_below_min_examples=skip_below_min_examples,
                    skip_above_max_examples=True,
                )

            # collate to python dict
            builder.map(Collater())

        pipeline = builder.prefetch(num_prefetch).and_return()

        return DataPipelineReader[HFBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=True,
        )


@final
class HFDatasetLoader(DatasetLoader[HFDataset]):
    def __call__(
        self,
        dataset_name_or_card: Union[str, AssetCard],
        *,
        force: bool = False,
        progress: bool = True,
    ) -> HFDataset:
        assert (
            isinstance(dataset_name_or_card, str)
        ), f"{type(dataset_name_or_card)} is not supported in HFDataset"
        return HFDataset(dataset_name_or_card)
        


# Usage:
# >>> from datasets import load_dataset()
# >>> from fairseq2.recipes.hf.datasets import hf_datasets
# >>> ds = load_dataset(DATASET_NAME, ...)
# >>> .... # Process ds
# >>> fs2_hf = hf_datasets("my_dataset")
# >>> hf_data_reader = fs2_hf.create_reader(dataset=ds, ....)
# >>> for data in hf_data_reader:
# >>>     # main logic
#
hf_datasets = HFDatasetLoader()
