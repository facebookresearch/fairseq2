# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, final

import torch
from typing_extensions import override

from fairseq2.assets import AssetCard, AssetError
from fairseq2.datasets.batching import Batching
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.typing import DataType


class SpeechDataset(ABC):
    """Represents a speech dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        gang: Gang,
        max_audio_len: int,
        batching: Batching,
        *,
        dtype: DataType = torch.float32,
        min_audio_len: int = 1,
        normalize_audio: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        drop_remainder: bool = False,
        sync_batches: bool = True,
        sync_mode: Literal["until_first", "until_last"] = "until_first",
        max_num_batches: int | None = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataReader[SequenceBatch]:
        """Create a dataset reader.

        :param split:
            The split to read.
        :param gang:
            The gang over which to shard the dataset.
        :param max_audio_len:
            The maximum audio length of each example. Examples longer than
            this value will be dropped.
        :param batching:
            The batching strategy for returned examples.
        :param dtype:
            The data type of the decoded audio sequences.
        :param min_audio_len:
            The minimum audio length of each example. Examples shorter than
            this value will be dropped.
        :param normalize_audio:
            If ``True``, normalizes audio to have zero mean and unit variance.
        :param example_shuffle_window:
            The size of the sliding window for shuffling examples. If ``1``, no
            shuffling is performed; if ``0``, true shuffling is performed by
            loading the entire dataset.
        :param batch_shuffle_window:
            The size of the sliding window for shuffling batches. If ``1``, no
            shuffling is performed; if ``0``, true shuffling is performed by
            loading the entire dataset.
        :param drop_remainder:
            If ``True``, drops the last set of batches if they have in total
            fewer examples than requested.
        :param sync_batches:
            If ``True``, ensures that each process in ``gang`` reads the same
            number of batches. Typically used when the amount of data to be read
            can vary per process (e.g. due to unbalanced sharding or non-static
            batching) and it is critical for each process to iterate over the
            same number of batches (e.g. during training).
        :param sync_mode:
            If ``until_first``, stops iteration when the first rank reaches end
            of data. If ``until_last``, stops iteration when the last rank
            reaches end of data; ranks that have already reached their end of
            data will return an empty list of batches.
        :param max_num_batches:
            The maximum number of batches to return.
        :param num_accumulate:
            The number of batches to accumulate in each iteration. Typically
            used with gradient accumulation during training.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param seed:
            The seed to initialize the random number generators used internally.
        :param extras:
            The extra parameters specific to the dataset implementation.
        """

    @abstractmethod
    def splits(self) -> set[str]:
        """Return the set of splits."""


load_speech_dataset = DelegatingDatasetLoader[SpeechDataset]()


@final
class GenericSpeechDataset(SpeechDataset):
    """Represents a generic manifest-based Speech dataset."""

    def __init__(self) -> None:
        pass

    @classmethod
    def from_path(cls, path: Path) -> GenericSpeechDataset:
        """Load a :class:`GenericSpeechDataset` from ``path``."""
        return GenericSpeechDataset()

    @override
    def create_reader(
        self,
        split: str,
        gang: Gang,
        max_audio_len: int,
        batching: Batching,
        *,
        dtype: DataType = torch.float32,
        min_audio_len: int = 1,
        normalize_audio: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        drop_remainder: bool = False,
        sync_batches: bool = True,
        sync_mode: Literal["until_first", "until_last"] = "until_first",
        max_num_batches: int | None = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        cached_fd_count: int = 1000,
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        """
        :param cached_fd_count:
            The maximum number of file descriptors to keep open while reading
            audio files.
        """
        raise RuntimeError("not supported yet.")

    @override
    def splits(self) -> set[str]:
        return set()


@final
class GenericSpeechDatasetLoader(AbstractDatasetLoader[GenericSpeechDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> GenericSpeechDataset:
        try:
            return GenericSpeechDataset.from_path(path)
        except RuntimeError as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex


load_generic_speech_dataset = GenericSpeechDatasetLoader()

load_speech_dataset.register("generic_speech", load_generic_speech_dataset)
