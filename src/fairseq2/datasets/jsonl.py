# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, final

from torch import Tensor
from typing_extensions import override

from fairseq2.data import (
    DataPipeline,
    read_sequence,
)
from fairseq2.data.text import read_text
from fairseq2.data.text.tokenizers import TextTokenEncoder
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DatasetLoadError,
    LengthBatching,
    SequenceBatch,
)
from fairseq2.datasets.text import TextDataset, TextReadOptions
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang
from fairseq2.nn import BatchLayout

# TODO: FIX, INFER
npc = 10


JSONL_DATASET_FAMILY: Final = "jsonl"


@final
class JsonlDataset(TextDataset):
    _name: str
    _files: Sequence[Path]

    def __init__(self, name: str, files: Sequence[Path]) -> None:
        self._name = name
        self._files = files

    @staticmethod
    def from_path(path: Path, name: str) -> JsonlDataset:
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

        return JsonlDataset(name, files)

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

        file_rank = gang.rank

        file_world_size = gang.size

        if len(self._files) < file_world_size:
            raise NotSupportedError(
                "The number of dataset files must be greater than or equal to the number of world size."
            )

        builder = read_sequence(self._files)

        if file_world_size > 1:
            builder.shard(file_rank, file_world_size, allow_uneven=True)

        def read_file(file: Path) -> DataPipeline:
            return read_text(file).map(json.loads, num_parallel_calls=1).and_return()

        builder.yield_from(read_file)

        # Tokenize.
        def encode(example: dict[str, Any]) -> Tensor:
            return text_encoder(example["text"])

        builder.map(encode, num_parallel_calls=1)

        batching = options.batching

        if isinstance(batching, LengthBatching):
            max_num_elements = batching.max_num_elements

            builder.pack(
                max_num_elements + 1, max_seq_len, truncate=True, pinned_memory=True
            )
        else:
            raise NotSupportedError(f"`{batching}` is not supported.")

        BatchLayout.compiled_max_seq_len = max_seq_len

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(options.num_prefetch)

        # Convert to `SequenceBatch`.
        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            seqs, seq_lens = example["seqs"], example["seq_lens"]

            return SequenceBatch(seqs, seq_lens, packed=True)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            self._name, "default", pipeline, gang, options
        )
