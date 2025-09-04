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
from typing import Any, Final, final

from torch import Tensor

from fairseq2.data.data_pipeline import (
    DataPipeline,
    read_sequence,
)
from fairseq2.data.text import read_text
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import DataPipelineReader, DataReader, SequenceBatch, SyncMode
from fairseq2.error import raise_operational_system_error
from fairseq2.gang import Gangs
from fairseq2.nn import BatchLayout

# TODO: FIX, INFER
npc = 10


LM_TRAIN_DATASET: Final = "lm_train"


@final
class LMTrainDataset:
    def __init__(self, files: Sequence[Path]) -> None:
        self._files = files

    def create_reader(
        self,
        tokenizer: Tokenizer,
        gangs: Gangs,
        *,
        max_seq_len: int,
        max_num_tokens: int,
        num_accumulate: int = 1,
        prefetch: int = 1,
        sync_ranks: bool = True,
    ) -> DataReader[SequenceBatch]:
        file_rank = gangs.dp.rank

        file_world_size = gangs.dp.size

        if len(self._files) < file_world_size:
            raise ValueError(
                "The number of dataset files must be greater than or equal to the number of world size."
            )

        builder = read_sequence(self._files)

        if file_world_size > 1:
            builder.shard(file_rank, file_world_size, allow_uneven=True)

        def read_file(file: Path) -> DataPipeline:
            return read_text(file).map(json.loads, num_parallel_calls=1).and_return()

        builder.yield_from(read_file)

        text_encoder = tokenizer.create_encoder(mode="default")

        # Tokenize.
        def encode(example: dict[str, Any]) -> Tensor:
            return text_encoder(example["text"])

        builder.map(encode, num_parallel_calls=1)

        # Pack
        builder.pack(max_num_tokens + 1, max_seq_len, truncate=True, pinned_memory=True)

        BatchLayout.compiled_max_seq_len = max_seq_len

        # Prefetch batches in background.
        builder.prefetch(prefetch)

        # Convert to `SequenceBatch`.
        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            seqs, seq_lens = example["seqs"], example["seq_lens"]

            return SequenceBatch(seqs, seq_lens, packed=True)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline,
            gangs,
            num_accumulate=num_accumulate,
            sync=sync_ranks,
            sync_mode=SyncMode.UNTIL_FIRST,
        )


@dataclass
class LMTrainDatasetConfig:
    path: Path = field(default_factory=Path)


def open_lm_train_dataset(config: LMTrainDatasetConfig) -> LMTrainDataset:
    path = config.path

    path = path.expanduser().resolve()

    if not path.is_dir():
        files = [path]
    else:
        try:
            files = [f for f in path.glob("**/*.chunk.*.jsonl") if not f.is_dir()]
        except OSError as ex:
            raise_operational_system_error(ex)

        files.sort()

    return LMTrainDataset(files)
