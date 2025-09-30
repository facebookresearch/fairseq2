# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast, final

from fairseq2.data.data_pipeline import (
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    read_sequence,
)
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import DataPipelineReader, SequenceBatch, SyncMode
from fairseq2.gang import Gangs

# TODO: FIX, INFER
npc = 10

TEXT_GEN_DATASET_FAMILY: Final = "text_generation"


# TODO: Work in progress!
@final
class TextGenDataset:
    def __init__(self, files: Sequence[Path]) -> None:
        self._files = [f.expanduser().resolve() for f in files]

    def create_reader(
        self, tokenizer: Tokenizer, gangs: Gangs, *, batch_size: int, prefetch: int = 1
    ) -> DataPipelineReader[SequenceBatch]:
        if len(self._files) == 1:
            builder = self._read_jsonl(self._files[0], tokenizer)
        else:
            pipelines = []

            for file in self._files:
                pipeline = self._read_jsonl(file, tokenizer).and_return()

                pipelines.append(pipeline)

            builder = DataPipeline.concat(pipelines)

        # Shard
        builder.shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)

        # Encode source texts.
        text_encoder = tokenizer.create_encoder(mode="prompt")

        def encode(example: dict[str, Any]) -> dict[str, Any]:
            id_ = example.get("id")

            source = example["src"]

            indices = text_encoder(source)

            return {"id": id_, "prompt": source, "indices": indices}

        builder.map(encode, num_parallel_calls=npc)

        # Bucket `batch_size` examples.
        builder.bucket(batch_size, drop_remainder=False)

        # Collate bucketed examples into a batch.
        pad_value = tokenizer.vocab_info.pad_idx
        if pad_value is None:
            pad_value = tokenizer.vocab_info.eos_idx

        collater = Collater(pad_value=pad_value)

        builder.map(collater, num_parallel_calls=npc)

        # Prefetch batches in background.
        builder.prefetch(prefetch)

        # Wrap examples with `SequenceBatch`.
        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            indices = cast(SequenceData, example["indices"])

            seqs, seq_lens = indices["seqs"], indices["seq_lens"]

            return SequenceBatch(seqs, seq_lens, example=example)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline, gangs, sync_mode=SyncMode.UNTIL_LAST
        )

    def _read_jsonl(self, path: Path, tokenizer: Tokenizer) -> DataPipelineBuilder:
        lines = []

        # TODO(balioglu): Do in C++.
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                lines.append(line)

        return read_sequence(lines).map(json.loads, num_parallel_calls=npc)


@dataclass
class TextGenDatasetConfig:
    paths: list[Path] = field(default_factory=list)


def open_text_gen_dataset(config: TextGenDatasetConfig) -> TextGenDataset:
    return TextGenDataset(config.paths)
