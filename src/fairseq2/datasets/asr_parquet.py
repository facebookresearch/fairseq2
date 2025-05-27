# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Final, List, Tuple, final

from typing_extensions import override

from fairseq2.data import CollateOptionsOverride, Collater, DataPipelineBuilder
from fairseq2.data.parquet import NamedColumns
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.datasets import DataPipelineReader, DataReader, UnknownSplitError
from fairseq2.datasets.asr import AsrDataset, GenericAsrDataset
from fairseq2.datasets.speech import GenericSpeechDataset, SpeechReadOptions
from fairseq2.datasets.speech_parquet import (
    GenericSpeechParquetDataset,
    ParquetDatasetInterface,
)
from fairseq2.gang import Gang
from fairseq2.logging import log
from fairseq2.models.seq2seq import Seq2SeqBatch

PARQUET_ASR_DATASET_FAMILY: Final = "generic_parquet_asr"


@dataclass
class DefaultASRSchema(NamedColumns):
    audio: str = "audio_bytes"
    length: str = "audio_size"
    text: str = "text"
    # size: str | None = "size"  # size of the audio in bytes : len(audio_bytes)
    extra_columns: List[str] | None = None


@final
class GenericAsrParquetDataset(ParquetDatasetInterface, AsrDataset):
    """Represents a generic parquet-based ASR dataset."""

    def build_example_reading_frontend(
        self,
        split: str,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
        options: SpeechReadOptions | None = None,
    ) -> Tuple[SpeechReadOptions, DataPipelineBuilder]:
        assert min_audio_len <= max_audio_len, "min_audio_len must be <= max_audio_len"

        if split not in self._splits:
            raise UnknownSplitError(self._name, split, self._splits)

        if options is None:
            options = SpeechReadOptions()

        options.batch_shuffle_window = min(
            options.batch_shuffle_window, self.max_num_batches
        )

        # FIXME: make it configurable, we need some upper bound to avoid OOM in cpu
        options.example_shuffle_window = min(
            options.example_shuffle_window, self.max_num_examples
        )

        log.info(
            f"Creating a reader for the <{split}> split of the <{self._name}>"
            f" dataset with the following options:/n {options}."
        )

        builder = GenericSpeechParquetDataset.get_example_loading_builder(
            self._dataset,
            options,
            split,
            columns=DefaultASRSchema(),
            seed=options.seed,
            rank=gang.rank,
            world_size=gang.size,
        )
        options.seed += gang.size
        # truncate length to max_audio_len
        builder = builder.filter(
            lambda x: (x["length"] >= min_audio_len) and (x["length"] <= max_audio_len)
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
    ) -> DataReader[Seq2SeqBatch]:
        options, builder = self.build_example_reading_frontend(
            split, gang, min_audio_len, max_audio_len, options
        )

        builder = GenericAsrParquetDataset.build_asr_main_pipeline(
            builder,
            options,
            tokenizer,
            min_audio_len=min_audio_len,
            max_audio_len=max_audio_len,
            gang=gang,
        )
        return DataPipelineReader[Seq2SeqBatch](
            self._name, split, builder.and_return(), gang, options, strict_state=False
        )

    @staticmethod
    def build_asr_main_pipeline(
        builder: DataPipelineBuilder,
        options: SpeechReadOptions,
        tokenizer: TextTokenizer,
        min_audio_len: int,
        max_audio_len: int,
        gang: Gang,
    ) -> DataPipelineBuilder:
        # shuffle examples in memory
        if options.example_shuffle_window != 1:
            builder = builder.shuffle(options.example_shuffle_window, seed=options.seed)
            options.seed += 1

        builder = GenericSpeechDataset.add_bucketing_pipeline(
            builder,
            options,
            max_audio_len=max_audio_len,
            min_audio_len=min_audio_len,
            seed=options.seed,
            columns="length",
        )
        builder = GenericAsrParquetDataset.build_parquet_audio_text_reading(
            builder, options, tokenizer, gang
        )
        return builder

    @staticmethod
    def build_parquet_audio_text_reading(
        builder: DataPipelineBuilder,
        options: SpeechReadOptions,
        tokenizer: TextTokenizer,
        gang: Gang,
    ) -> DataPipelineBuilder:
        builder = GenericSpeechParquetDataset.add_audio_decoding(builder, options)
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
