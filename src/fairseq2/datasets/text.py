# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final, cast, final

from typing_extensions import override

from fairseq2.data import (
    Collater,
    DataPipeline,
    SequenceData,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.data.text import read_text
from fairseq2.data.text.tokenizers import TextTokenEncoder
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DataReadOptions,
    DatasetHubAccessor,
    DatasetLoadError,
    LengthBatching,
    StaticBatching,
)
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import Device


@dataclass(kw_only=True)
class TextReadOptions(DataReadOptions):
    pass


class TextDataset(ABC):
    """Represents a text dataset."""

    @abstractmethod
    def create_reader(
        self,
        text_encoder: TextTokenEncoder,
        pad_idx: int | None,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: TextReadOptions | None = None,
    ) -> DataReader[SequenceBatch]:
        """Create a dataset reader.

        :param text_encoder:
            The text token encoder.
        :param pad_idx:
            The index of the PAD symbol in the vocabulary of ``text_encoder``.
        :param gang:
            The gang over which to shard the dataset.
        :param min_seq_len:
            The minimum sequence length of each example. Examples shorter than
            this value will be dropped.
        :param max_seq_len:
            The maximum sequence length of each example. Examples longer than
            this value will be dropped.
        :param options:
            The read options.
        """


# TODO: FIX, INFER
npc = 10


GENERIC_TEXT_DATASET_FAMILY: Final = "generic_text"


@final
class GenericTextDataset(TextDataset):
    """Represents a generic file-based text dataset."""

    _name: str
    _files: Sequence[Path]

    def __init__(self, name: str, files: Sequence[Path]) -> None:
        """
        :param data_dir:
            The list of text files that represent the dataset.
        """
        self._name = name
        self._files = files

    @staticmethod
    def from_path(path: Path, name: str) -> GenericTextDataset:
        path = path.expanduser().resolve()

        if not path.is_dir():
            files = [path]
        else:
            try:
                files = [f for f in path.glob("**/*.txt") if not f.is_dir()]
            except OSError as ex:
                raise DatasetLoadError(
                    name, f"The text files under the '{path}' directory of the '{name}' dataset cannot be retrieved. See the nested exception for details."  # fmt: skip
                ) from ex

            files.sort()

        return GenericTextDataset(name, files)

    @override
    def create_reader(
        self,
        text_encoder: TextTokenEncoder,
        pad_idx: int | None,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: TextReadOptions | None = None,
    ) -> DataReader[SequenceBatch]:
        if options is None:
            options = TextReadOptions()

        seed = options.seed

        if len(self._files) == 1:
            builder = read_text(self._files[0], key="text", rtrim=True)
        else:
            builder = read_sequence(self._files)

            def read_file(file: Path) -> DataPipeline:
                return read_text(file, key="text", rtrim=True).and_return()

            builder.yield_from(read_file)

        # Shuffle examples. Must be consistent across all processes.
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        def encode(example: dict[str, Any]) -> dict[str, Any]:
            example["indices"] = text_encoder(example["text"])

            return example

        builder.map(encode, num_parallel_calls=npc)

        batching = options.batching

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
                drop_remainder=options.drop_remainder,
            )
        elif isinstance(batching, StaticBatching):
            # Filter out out-of-range examples.
            def skip(example: dict[str, Any]) -> bool:
                seq_len = len(example["indices"])

                return seq_len >= min_seq_len and seq_len <= max_seq_len

            builder.filter(skip)

            # Bucket `batch_size` examples.
            builder.bucket(batching.batch_size, drop_remainder=options.drop_remainder)
        else:
            raise NotSupportedError(f"`{batching}` is not supported.")

        # Shuffle buckets.
        if options.batch_shuffle_window != 1:
            builder.shuffle(options.batch_shuffle_window, seed)

        seed += 1

        # Collate bucketed examples into a batch.
        collater = Collater(pad_value=pad_idx)

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(options.num_prefetch)

        f = partial(self._to_batch, device=gang.device)

        pipeline = builder.map(f).and_return()

        return DataPipelineReader[SequenceBatch](
            self._name, "default", pipeline, gang, options
        )

    @staticmethod
    def _to_batch(example: dict[str, Any], device: Device) -> SequenceBatch:
        data = cast(SequenceData, example["indices"])

        seqs, padding_mask = get_seqs_and_padding_mask(data, device)

        return SequenceBatch(seqs, padding_mask, example=example)


get_text_dataset_hub = DatasetHubAccessor(TextDataset)
