# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, Final, cast

from typing_extensions import override

from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    read_sequence,
)
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DatasetHubAccessor,
    UnknownSplitError,
)
from fairseq2.datasets.speech import (
    GenericSpeechDataset,
    ManifestDatasetInterface,
    SpeechReadOptions,
)
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import Device


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
        options: SpeechReadOptions | None = None,
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


GENERIC_ASR_DATASET_FAMILY: Final = "generic_asr"

get_asr_dataset_hub = DatasetHubAccessor(AsrDataset)


class GenericAsrDataset(ManifestDatasetInterface, AsrDataset):
    """Represents a generic manifest-based ASR dataset."""

    @staticmethod
    def to_batch(example: Dict[str, Any], device: Device | None = None) -> Seq2SeqBatch:
        source_data = cast(SequenceData, example["audio_feature"])
        target_data = cast(SequenceData, example["text"])

        source_seqs, source_padding_mask = get_seqs_and_padding_mask(
            source_data, device=device
        )
        target_seqs, target_padding_mask = get_seqs_and_padding_mask(
            target_data, device=device
        )

        return Seq2SeqBatch(
            source_seqs,
            source_padding_mask,
            target_seqs,
            target_padding_mask,
            example,
        )

    @override
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
        options: SpeechReadOptions | None = None,
    ) -> DataPipelineReader[Seq2SeqBatch]:
        """
        :param cached_fd_count:
            The maximum number of file descriptors to keep open while reading
            audio files.
        """
        if split not in self._splits:
            raise UnknownSplitError(self._name, split, self._splits)

        if options is None:
            options = SpeechReadOptions()
        npc = options.npc
        seed = options.seed

        audio_dir = GenericSpeechDataset._retrieve_data_directory(
            self._manifest_dir, self._name, split
        )
        builder = self._read_manifest(split)

        # Shuffle examples. Must be consistent across all processes.
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank
        # Bucketize examples by audio length.
        builder = GenericSpeechDataset.add_bucketing_pipeline(
            builder, options, max_audio_len, min_audio_len, seed, "audio_size"
        )
        # Read audios
        seed += 1
        builder = GenericSpeechDataset.add_audio_decoding(builder, options, audio_dir)
        builder = GenericSpeechDataset.audio_post_process(
            builder, options, GenericSpeechDataset.rename_feature
        )

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

        pipeline = builder.map(partial(self.to_batch, device=gang.device)).and_return()

        return DataPipelineReader[Seq2SeqBatch](
            self._name, split, pipeline, gang, options
        )

    def _read_manifest(self, split: str) -> DataPipelineBuilder:
        def read_tsv_file() -> DataPipelineBuilder:
            tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")

            builder = read_text(tsv_file, rtrim=True, memory_map=True)

            builder.skip(1)  # Path to the data directory.

            field_splitter = StrSplitter(names=["audio", "audio_size"])

            builder.map(field_splitter)

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
