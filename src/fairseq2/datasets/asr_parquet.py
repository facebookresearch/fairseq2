# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Final, List, final

from typing_extensions import override

from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
)
from fairseq2.data.parquet import (
    NamedColumns,
)
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    UnknownSplitError,
)
from fairseq2.datasets.asr import AsrDataset, GenericAsrDataset
from fairseq2.datasets.speech import (
    GenericSpeechDataset,
    SpeechReadOptions,
)
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
        assert min_audio_len <= max_audio_len, "min_audio_len must be <= max_audio_len"

        if split not in self._splits:
            raise UnknownSplitError(self._name, split, self._splits)

        if options is None:
            options = SpeechReadOptions()
        npc = options.npc
        seed = options.seed

        options.batch_shuffle_window = min(
            options.batch_shuffle_window, self.max_num_batches
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
            seed=seed,
            rank=gang.rank,
            world_size=gang.size,
        )
        seed += gang.size
        # truncate length to max_audio_len
        builder = builder.filter(
            lambda x: (int(x["length"]) >= min_audio_len)
            and (int(x["length"]) <= max_audio_len)
        )

        # shuffle examples in memory
        if options.example_shuffle_window != 1:
            # FIXME: make it configurable, we need some upper bound to avoid OOM in cpu
            example_shuffle_window = min(
                options.example_shuffle_window, self.max_num_examples
            )
            builder = builder.prefetch(int(2 * example_shuffle_window))
            builder = builder.shuffle(example_shuffle_window, seed=seed)
            seed += 1

        builder = GenericSpeechDataset.add_bucketing_pipeline(
            builder,
            options,
            max_audio_len=max_audio_len,
            min_audio_len=min_audio_len,
            seed=seed,
            columns="length",
        )
        seed += 1

        # Read audios
        seed += 1
        builder = GenericSpeechParquetDataset.add_audio_decoding(builder, options)
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

        pipeline = builder.map(
            partial(GenericAsrDataset.to_batch, device=gang.device)
        ).and_return()

        return DataPipelineReader[Seq2SeqBatch](
            self._name, split, pipeline, gang, options, strict_state=False
        )
