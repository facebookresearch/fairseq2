# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast, final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetError
from fairseq2.data import (
    Collater,
    DataPipeline,
    SequenceData,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.data.text import TextTokenEncoder, read_text
from fairseq2.datasets.batching import Batching, LengthBatching, StaticBatching
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import Device


class TextDataset(ABC):
    """Represents a text dataset."""

    @abstractmethod
    def create_reader(
        self,
        text_encoder: TextTokenEncoder,
        pad_idx: int | None,
        gang: Gang,
        max_seq_len: int,
        batching: Batching,
        *,
        min_seq_len: int = 1,
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

        :param text_encoder:
            The text token encoder.
        :param pad_idx:
            The index of the PAD symbol in the vocabulary of ``text_encoder``.
        :param gang:
            The gang over which to shard the dataset.
        :param max_seq_len:
            The maximum sequence length of each example. Examples longer than
            this value will be dropped.
        :param batching:
            The batching strategy for returned examples.
        :param min_seq_len:
            The minimum sequence length of each example. Examples shorter than
            this value will be dropped.
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


load_text_dataset = DelegatingDatasetLoader[TextDataset]()


# TODO: FIX, INFER
npc = 10


@final
class GenericTextDataset(TextDataset):
    """Represents a generic file-based text dataset."""

    _files: Sequence[Path]

    def __init__(self, files: Sequence[Path]) -> None:
        """
        :param data_dir:
            The list of text files that represent the dataset.
        """
        self._files = files

    @staticmethod
    def from_path(path: Path) -> GenericTextDataset:
        """Load a :class:`GenericTextDataset` from ``path``."""
        path = path.expanduser().resolve()

        if not path.is_dir():
            files = [path]
        else:
            try:
                files = [f for f in path.glob("**/*.txt") if not f.is_dir()]
            except OSError as ex:
                raise RuntimeError(
                    f"The text files under {path} cannot be retrieved. See nested exception for details."
                ) from ex

            files.sort()

        return GenericTextDataset(files)

    @override
    def create_reader(
        self,
        text_encoder: TextTokenEncoder,
        pad_idx: int | None,
        gang: Gang,
        max_seq_len: int,
        batching: Batching,
        *,
        min_seq_len: int = 1,
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
        if len(self._files) == 1:
            builder = read_text(self._files[0], key="text", rtrim=True)
        else:
            builder = read_sequence(self._files)

            def read_file(file: Path) -> DataPipeline:
                return read_text(file, key="text", rtrim=True).and_return()

            builder.yield_from(read_file)

        # Shuffle examples. Must be consistent across all processes.
        if example_shuffle_window != 1:
            builder.shuffle(example_shuffle_window, seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        def encode(example: dict[str, Any]) -> dict[str, Any]:
            example["indices"] = text_encoder(example["text"])

            return example

        builder.map(encode, num_parallel_calls=npc)

        if isinstance(batching, LengthBatching):
            # Bucket by the length of the sequence.
            bucket_sizes = create_bucket_sizes(
                max_seq_len=max_seq_len,
                min_seq_len=min_seq_len,
                max_num_elements=batching.max_num_elements,
            )

            builder.bucket_by_length(
                bucket_sizes,
                selector="indices",
                min_data_len=min_seq_len,
                skip_below_min_examples=True,
                skip_above_max_examples=True,
                drop_remainder=drop_remainder,
            )
        elif isinstance(batching, StaticBatching):
            # Filter out out-of-range examples.
            def skip(example: dict[str, Any]) -> bool:
                seq_len = len(example["indices"])

                return seq_len >= min_seq_len and seq_len <= max_seq_len

            builder.filter(skip)

            # Bucket `batch_size` examples.
            builder.bucket(batching.batch_size, drop_remainder=drop_remainder)
        else:
            raise RuntimeError(f"`{batching}` is not supported.")

        # Shuffle buckets.
        if batch_shuffle_window != 1:
            builder.shuffle(batch_shuffle_window, seed)

        seed += 1

        # Collate bucketed examples into a batch.
        collater = Collater(pad_value=pad_idx)

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(num_prefetch)

        f = partial(self._to_batch, device=gang.device)

        pipeline = builder.map(f).and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=drop_remainder,
            sync_batches=sync_batches,
            sync_mode=sync_mode,
        )

    @staticmethod
    def _to_batch(example: dict[str, Any], device: Device) -> SequenceBatch:
        data = cast(SequenceData, example["indices"])

        seqs, padding_mask = get_seqs_and_padding_mask(data, device)

        return SequenceBatch(seqs, padding_mask, example=example)


@final
class GenericTextDatasetLoader(AbstractDatasetLoader[GenericTextDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> GenericTextDataset:
        try:
            return GenericTextDataset.from_path(path)
        except RuntimeError as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex


load_generic_text_dataset = GenericTextDatasetLoader()

load_text_dataset.register("generic_text", load_generic_text_dataset)
