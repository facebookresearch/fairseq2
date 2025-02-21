# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast, final

import torch
from torch import Tensor
from torch.nn.functional import layer_norm
from typing_extensions import override

from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    FileMapper,
    SequenceData,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.data.audio import AudioDecoder
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DataReadError,
    DataReadOptions,
    DatasetHubAccessor,
    DatasetLoadError,
    LengthBatching,
    StaticBatching,
    UnknownSplitError,
)
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import DataType


@dataclass(kw_only=True)
class AsrReadOptions(DataReadOptions):
    dtype: DataType = torch.float32
    """The data type of the decoded audio sequences."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""


class AsrDataset(ABC):
    """Represents an automatic speech recognition dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
        options: AsrReadOptions | None = None,
    ) -> DataReader[Seq2SeqBatch]:
        """Create a dataset reader.

        :param split:
            The split to read.
        :param tokenizer:
            The tokenizer to encode target text.
        :param gang:
            The gang over which to shard the dataset.
        :param min_audio_len:
            The minimum audio length of each example. Examples shorter than this
            value will be dropped.
        :param max_audio_len:
            The maximum audio length of each example. Examples longer than this
            value will be dropped.
        :param options:
            The read options.
        """

    @abstractmethod
    def splits(self) -> set[str]:
        """Return the set of splits."""


# TODO: FIX, INFER
npc = 10


GENERIC_ASR_DATASET_FAMILY: Final = "generic_asr"


# TODO: Work in progress!
@final
class GenericAsrDataset(AsrDataset):
    """Represents a generic manifest-based ASR dataset."""

    _name: str
    _manifest_dir: Path
    _splits: set[str]

    def __init__(self, name: str, manifest_dir: Path, splits: set[str]) -> None:
        """
        :param manifest_dir:
            The directory under which the manifest files resides.
        :param splits:
            The available splits.
        """
        self._name = name
        self._manifest_dir = manifest_dir
        self._splits = splits

    @staticmethod
    def from_path(path: Path, name: str) -> GenericAsrDataset:
        path = path.expanduser().resolve()

        if not path.is_dir():
            return GenericAsrDataset(name, manifest_dir=path.parent, splits={path.stem})

        try:
            splits = {f.stem for f in path.glob("*.tsv")}
        except OSError as ex:
            raise DatasetLoadError(
                name, f"The splits under the '{path}' directory of the '{name}' dataset cannot be determined. See the nested exception for details."  # fmt: skip
            ) from ex

        return GenericAsrDataset(name, path, splits)

    @override
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
        options: AsrReadOptions | None = None,
    ) -> DataPipelineReader[Seq2SeqBatch]:
        """
        :param cached_fd_count:
            The maximum number of file descriptors to keep open while reading
            audio files.
        """
        if split not in self._splits:
            raise UnknownSplitError(self._name, split, self._splits)

        if options is None:
            options = AsrReadOptions()

        seed = options.seed

        audio_dir = self._retrieve_data_directory(split)

        builder = self._read_manifest(split)

        # Shuffle examples. Must be consistent across all processes.
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        batching = options.batching

        if isinstance(batching, LengthBatching):
            # Bucket by the audio length.
            bucket_sizes = create_bucket_sizes(
                min_seq_len=min_audio_len,
                max_seq_len=max_audio_len,
                max_num_elements=batching.max_num_elements,
                num_seqs_multiple_of=8,
            )

            builder.bucket_by_length(
                bucket_sizes,
                selector="audio_size",
                min_data_len=min_audio_len,
                skip_below_min_examples=True,
                skip_above_max_examples=True,
                drop_remainder=options.drop_remainder,
            )
        elif isinstance(batching, StaticBatching):
            # Filter out out-of-range audios.
            def skip(example: dict[str, object]) -> bool:
                audio_len = cast(int, example["audio_size"])

                return audio_len >= min_audio_len and audio_len <= max_audio_len

            builder.filter(skip)

            # Bucket `batch_size` examples.
            builder.bucket(batching.batch_size, drop_remainder=options.drop_remainder)
        else:
            raise NotSupportedError(f"`{batching}` is not supported.")

        # Shuffle buckets.
        if options.batch_shuffle_window != 1:
            builder.shuffle(options.batch_shuffle_window, seed)

        seed += 1

        # Memory map audio files.
        cached_fd_count = options.extras.get("cached_fd_count", 1)
        if not isinstance(cached_fd_count, int):
            raise TypeError(
                f"`options.extras['cached_fd_count']` must be of type `int`, but is of type `{type(cached_fd_count)}` instead."
            )

        file_mapper = FileMapper(audio_dir, cached_fd_count=cached_fd_count)

        builder.map(file_mapper, selector="[*].audio")

        # Decode audio.
        audio_decoder = AudioDecoder(
            dtype=torch.float32 if options.normalize_audio else options.dtype
        )

        builder.map(audio_decoder, selector="[*].audio.data")

        # TODO(balioglu): Check/adjust sample size

        # Normalize audio if requested.
        def normalize(waveform: Tensor) -> Tensor:
            with torch.no_grad():
                waveform = layer_norm(waveform, waveform.shape)

            return waveform.to(options.dtype)

        if options.normalize_audio:
            builder.map(normalize, selector="[*].audio.data.waveform")

        # Tokenize target text.
        text_encoder = tokenizer.create_encoder()

        builder.map(text_encoder, selector="[*].text", num_parallel_calls=npc)

        # Collate bucketed examples into a batch.
        text_collate_opts = CollateOptionsOverride(
            "text", pad_value=tokenizer.vocab_info.pad_idx
        )

        collater = Collater(pad_value=0, overrides=[text_collate_opts])

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(options.num_prefetch)

        # Wrap examples with `Seq2SeqBatch`.
        def to_batch(example: dict[str, Any]) -> Seq2SeqBatch:
            source_data = cast(SequenceData, example["audio"]["data"]["waveform"])
            target_data = cast(SequenceData, example["text"])

            source_seqs, source_padding_mask = get_seqs_and_padding_mask(
                source_data, gang.device
            )
            target_seqs, target_padding_mask = get_seqs_and_padding_mask(
                target_data, gang.device
            )

            return Seq2SeqBatch(
                source_seqs,
                source_padding_mask,
                target_seqs,
                target_padding_mask,
                example,
            )

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[Seq2SeqBatch](
            self._name, split, pipeline, gang, options
        )

    def _retrieve_data_directory(self, split: str) -> Path:
        manifest_file = self._manifest_dir.joinpath(f"{split}.tsv")

        try:
            with manifest_file.open() as fp:
                line = fp.readline().rstrip()
        except OSError as ex:
            raise DataReadError(
                self._name, split, f"The {manifest_file} manifest file cannot be read. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            return Path(line)
        except ValueError:
            raise DataReadError(
                self._name, split, f"The first line of the '{manifest_file}' manifest file must point to a data directory."  # fmt: skip
            ) from None

    def _read_manifest(self, split: str) -> DataPipelineBuilder:
        def read_tsv_file() -> DataPipelineBuilder:
            tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")

            builder = read_text(tsv_file, rtrim=True, memory_map=True)

            builder.skip(1)  # Path to the data directory.

            field_splitter = StrSplitter(names=["audio", "audio_size"])

            builder.map(field_splitter, num_parallel_calls=npc)

            return builder

        def read_wrd_file() -> DataPipelineBuilder:
            wrd_file = self._manifest_dir.joinpath(f"{split}.wrd")

            return read_text(wrd_file, key="text", rtrim=True, memory_map=True)

        tsv_pipeline = read_tsv_file().and_return()
        wrd_pipeline = read_wrd_file().and_return()

        builder = DataPipeline.zip([tsv_pipeline, wrd_pipeline], flatten=True)

        # Cast audio size to integer.
        builder.map(int, selector="audio_size")

        # TODO(balioglu): Use `cache()` op.
        manifest = list(builder.and_return())

        return read_sequence(manifest)

    @override
    def splits(self) -> set[str]:
        return self._splits


get_asr_dataset_hub = DatasetHubAccessor(AsrDataset)
