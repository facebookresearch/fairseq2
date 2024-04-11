# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar

from fairseq2.assets import default_asset_store, default_download_manager
from fairseq2.data.text import TextTokenizer
from fairseq2.datasets.data_reader import DataReader
from fairseq2.datasets.loader import (
    DatasetFactory,
    DatasetLoader,
    DelegatingDatasetLoader,
    StandardDatasetLoader,
)
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.typing import DataType


class AsrDataset(ABC):
    """Represents an automatic speech recognition dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_audio_len: int,
        max_num_elements: int,
        *,
        dtype: Optional[DataType] = None,
        min_audio_len: int = 1,
        shuffle_window_size: int = 0,
        num_repeats: Optional[int] = 1,
        num_prefetch: int = 0,
        num_accumulate: int = 1,
    ) -> DataReader[Seq2SeqBatch]:
        """Create a dataset reader.

        :param split:
            The split to read.
        :param tokenizer:
            The tokenizer to encode target text.
        :param gang:
            The gang over which to shard the dataset.
        :param max_audio_len:
            The maximum audio length of each example. Examples longer than
            this value will be cropped.
        :param max_num_elements:
            The maximum number of elements in each batch.
        :param dtype:
            The data type of the decoded audio sequences.
        :param min_audio_len:
            The minimum audio length of each example. Examples shorter than
            this value will be dropped.
        :param shuffle_window_size:
            The size of the streaming shuffle window.
        :param num_repeats:
            The dataset will be repeatedly read this many times. If ``None``, it
            will be read indefinitely.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param num_accumulate:
            The number of batches to accumulate in each iteration. Typically
            used with gradient accumulation during training.
        """

    @abstractmethod
    def splits(self) -> List[str]:
        """Return the list of splits."""


load_asr_dataset = DelegatingDatasetLoader[AsrDataset](default_asset_store)


AsrDatasetT = TypeVar("AsrDatasetT", bound=AsrDataset)


def setup_asr_dataset(
    family: str, factory: DatasetFactory[AsrDatasetT]
) -> DatasetLoader[AsrDatasetT]:
    """Set up an automatic speech recognition dataset.

    :param family:
        The name of the dataset family.
    :param factory:
        The factory to construct datasets.
    """
    loader = StandardDatasetLoader[AsrDatasetT](
        default_asset_store, default_download_manager, factory
    )

    load_asr_dataset.register_loader(family, loader)

    return loader
