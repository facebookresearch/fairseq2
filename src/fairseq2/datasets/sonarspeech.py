# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, cast, Dict, Final, Tuple

import torch

import torchaudio
from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    MemoryBlock,
    read_sequence,
    SequenceData,
)

from fairseq2.data.audio import AudioDecoder, AudioDecoderOutput
from fairseq2.data.text import read_text, StrSplitter
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DatasetHubAccessor,
    UnknownSplitError,
)
from fairseq2.datasets.asr import AsrDataset
from fairseq2.datasets.speech import (
    GenericSpeechDataset,
    ManifestDatasetInterface,
    SpeechReadOptions,
)
from fairseq2.datasets.speech_parquet import ParquetDatasetInterface
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch, SonarSpeechSeq2SeqBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import Device
from typing_extensions import override

GENERIC_SONAR_SPEECH_DATASET_FAMILY: Final = "generic_sonar_speech"

get_asr_dataset_hub = DatasetHubAccessor(AsrDataset)


class GenericSonarSpeechDataset(ManifestDatasetInterface, AsrDataset):
    """Represents a generic manifest-based ASR dataset."""

    @staticmethod
    def add_audio_decoding(
        builder: DataPipelineBuilder, options: SpeechReadOptions
    ) -> DataPipelineBuilder:

        audio_decoder = AudioDecoder(
            dtype=torch.float32 if options.normalize_audio else options.dtype
        )

        def decoded_audio(_bytes: NDArray[np.int8]) -> Dict[str, AudioDecoderOutput]:
            return {"data": audio_decoder(MemoryBlock(_bytes.tobytes()))}

        builder.map(decoded_audio, selector="[*].audio", num_parallel_calls=options.npc)

        return builder

    @staticmethod
    def to_batch(
        example: Dict[str, Any], device: Device | None = None
    ) -> SonarSpeechSeq2SeqBatch:
        source_data = cast(SequenceData, example["audio_feature"])
        target_data = cast(SequenceData, example["text"])
        target_embeddings = example["text_sonar_emb"]

        source_seqs, source_padding_mask = get_seqs_and_padding_mask(
            source_data, device=device
        )
        target_seqs, target_padding_mask = get_seqs_and_padding_mask(
            target_data, device=device
        )
        target_embeddings, _ = get_seqs_and_padding_mask(
            target_embeddings, device=device
        )
        if target_embeddings.ndim == 3:
            target_embeddings = target_embeddings.squeeze(1)

        # print(f"source padding mask: {source_padding_mask}")
        return SonarSpeechSeq2SeqBatch(
            source_seqs,
            source_padding_mask,
            target_seqs,
            target_padding_mask,
            example,
            target_embeddings,
        )

    def build_example_reading_frontend(
        self,
        split: str,
        gang: Gang,
        options: SpeechReadOptions | None = None,
    ) -> Tuple[SpeechReadOptions, DataPipelineBuilder]:
        if split not in self._splits:
            raise UnknownSplitError(self._name, split, self._splits)

        if options is None:
            options = SpeechReadOptions()

        builder = self._read_manifest(split)

        # Shuffle examples. Must be consistent across all processes.
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, options.seed)

        options.seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        options.seed += gang.rank
        audio_dir = GenericSpeechDataset._retrieve_data_directory(
            self._manifest_dir, self._name, split
        )
        if audio_dir is not None:
            builder = builder.map(
                lambda audio_path: str(audio_dir.joinpath(audio_path)), selector="audio"
            )

        return options, builder

    @override
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
        options: SpeechReadOptions | None = None,
    ) -> DataPipelineReader[SonarSpeechSeq2SeqBatch]:
        options, builder = self.build_example_reading_frontend(split, gang, options)
        builder = GenericAsrDataset.build_asr_main_pipeline(
            builder,
            options,
            tokenizer,
            gang=gang,
            min_audio_len=min_audio_len,
            max_audio_len=max_audio_len,
        )
        return DataPipelineReader[SonarSpeechSeq2SeqBatch](
            self._name, split, builder.and_return(), gang, options
        )

    @staticmethod
    def build_asr_main_pipeline(
        builder: DataPipelineBuilder,
        options: SpeechReadOptions,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
    ) -> DataPipelineBuilder:

        # Bucketize examples by audio length.
        builder = GenericSpeechDataset.add_bucketing_pipeline(
            builder,
            options,
            max_audio_len,
            min_audio_len,
            seed=options.seed,
            columns="audio_size",
        )
        # Read audios
        options.seed += 1
        builder = GenericSpeechDataset.add_audio_decoding(
            builder, options, audio_dir=None
        )
        builder = GenericSpeechDataset.audio_post_process(
            builder, options, GenericSpeechDataset.rename_feature
        )

        # Tokenize target text.
        text_encoder = tokenizer.create_encoder()
        builder.map(text_encoder, selector="[*].text", num_parallel_calls=options.npc)

        # Collate bucketed examples into a batch.
        text_collate_opts = CollateOptionsOverride(
            "text", pad_value=tokenizer.vocab_info.pad_idx
        )

        collater = Collater(pad_value=0, overrides=[text_collate_opts])

        builder.map(collater, num_parallel_calls=options.npc)

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(options.num_prefetch)

        # Wrap examples with `Seq2SeqBatch`.

        builder = builder.map(partial(GenericAsrDataset.to_batch, device=gang.device))

        return builder

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
        manifest = list(builder.and_return())

        return read_sequence(manifest)
