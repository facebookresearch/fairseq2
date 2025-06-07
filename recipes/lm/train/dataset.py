# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, final

from torch import Tensor

from fairseq2.data.data_pipeline import (
    DataPipeline,
    read_sequence,
)
from fairseq2.data.text import read_text
from fairseq2.data.tokenizers import TokenEncoder
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DataReadOptions,
    DatasetLoadError,
    SequenceBatch,
)
from fairseq2.dependency import DependencyResolver
from fairseq2.gang import Gang
from fairseq2.nn import BatchLayout

# TODO: FIX, INFER
npc = 10


JSONL_TEXT_DATASET_FAMILY: Final = "jsonl_text"


@dataclass(kw_only=True)
class TextReadOptions(DataReadOptions):
    pass


@final
class JsonlTextDataset:
    _name: str
    _files: Sequence[Path]

    def __init__(self, name: str, files: Sequence[Path]) -> None:
        self._name = name
        self._files = files

    @staticmethod
    def from_path(
        resolver: DependencyResolver, path: Path, name: str
    ) -> JsonlTextDataset:
        path = path.expanduser().resolve()

        if not path.is_dir():
            files = [path]
        else:
            try:
                files = [f for f in path.glob("**/*.chunk.*.jsonl") if not f.is_dir()]
            except OSError as ex:
                raise DatasetLoadError(
                    name, f"The text files under the '{path}' directory of the '{name}' dataset cannot be retrieved. See the nested exception for details."  # fmt: skip
                ) from ex

            files.sort()

        return JsonlTextDataset(name, files)

    def create_reader(
        self,
        text_encoder: TokenEncoder,
        pad_idx: int | None,
        gang: Gang,
        # TODO: remove?
        min_seq_len: int,
        max_seq_len: int,
        options: TextReadOptions | None = None,
    ) -> DataReader[SequenceBatch]:
        if options is None:
            options = TextReadOptions()

        seed = options.seed + gang.rank

        files = self._files
        file_rank = gang.rank
        file_world_size = gang.size

        builder = read_sequence(files)

        if file_world_size > 1:
            builder.shard(file_rank, file_world_size, allow_uneven=True)

        def read_file(file: Path) -> DataPipeline:
            return read_text(file).map(json.loads, num_parallel_calls=npc).and_return()

        builder.yield_from(read_file)

        #        # Shuffle examples.
        #        if options.example_shuffle_window != 1:
        #            builder.shuffle(options.example_shuffle_window, seed)

        seed += 1

        # Tokenize.
        def encode(example: dict[str, Any]) -> Tensor:
            return text_encoder(example["text"])

        builder.map(encode, num_parallel_calls=npc)
        #        builder.map(text_encoder, selector="text", num_parallel_calls=npc)

        bsz = 2
        sl = 8192
        #        builder.pack((2 * 8192) + 1, 8192, truncate=True, pinned_memory=True)
        builder.pack((bsz * sl) + 1, sl, truncate=True, pinned_memory=True)

        BatchLayout.compiled_max_seq_len = sl

        #        # Shuffle batches.
        #        if options.batch_shuffle_window != 1:
        #            builder.shuffle(options.batch_shuffle_window, seed)

        seed += 1

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        #        builder.prefetch(options.num_prefetch)

        # Convert to `SequenceBatch`.
        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            seqs, seq_lens = example["seqs"], example["seq_lens"]

            return SequenceBatch(seqs, seq_lens, packed=True)

        pipeline = builder.map(to_batch).and_return()

        #        pipeline = builder.prefetch(1000).and_return()
        buf = []

        it = iter(pipeline)

        for _ in range(1000):
            buf.append(next(it))

        pipeline = read_sequence(buf).and_return()

        return DataPipelineReader[SequenceBatch](
            self._name, "default", pipeline, gang, options
        )
