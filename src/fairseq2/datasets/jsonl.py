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

from fairseq2.data import DataPipeline, DataPipelineBuilder, read_sequence
from fairseq2.data.text import read_text
from fairseq2.data.text.tokenizers import TextTokenEncoder
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DatasetLoadError,
    LengthBatching,
    SequenceBatch,
)
from fairseq2.datasets.text import GenericTextDataset, TextDataset, TextReadOptions
from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang
from fairseq2.logging import log
from fairseq2.nn import BatchLayout

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
        split: str | None = None,
    ) -> DataReader[SequenceBatch]:
        if options is None:
            options = TextReadOptions()

        file_rank = gang.rank
        file_world_size = gang.size

        text_column_name = options.extras.get("text_column_name", "text")
        assert isinstance(text_column_name, str)

        if min_seq_len > 0:
            log.warning(
                f"The `min_seq_len={min_seq_len}`  is ignored because of packing."
            )

        split_pattern = options.extras.get("split_pattern", None)
        split_files = GenericTextDataset.filter_split(
            self._files,
            split,
            extention="jsonl",
            split_pattern=split_pattern,  # type: ignore[arg-type]
        )

        if len(split_files) < file_world_size:
            raise NotSupportedError(
                "The number of dataset files must be greater than or equal to the number of world size."
            )

        builder = read_sequence(split_files)

        if file_world_size > 1:
            builder.shard(file_rank, file_world_size, allow_uneven=True)

        def read_file(file: Path) -> DataPipeline:
            return read_text(file).map(json.loads).and_return()

        builder.yield_from(read_file)

        pipeline = JsonlDataset.build_pipeline_backend(
            builder,
            options,
            text_encoder,
            pad_idx=pad_idx,
            max_seq_len=max_seq_len,
            text_column_name=text_column_name,
            device=gang.device,
        )
        return DataPipelineReader[SequenceBatch](
            self._name, "default", pipeline, gang, options
        )

    @staticmethod
    def build_pipeline_backend(
        builder: DataPipelineBuilder,
        options: TextReadOptions,
        text_encoder: TextTokenEncoder,
        max_seq_len: int,
        pad_idx: int | None,
        text_column_name: str,
        device: Device,
    ) -> DataPipeline:
        if pad_idx is None:
            pad_idx = 0

        seed = options.seed
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed)

        # Tokenize.
        def encode(example: dict[str, Any]) -> Tensor:
            return text_encoder(example[text_column_name])

        builder.map(encode)

        batching = options.batching

        if isinstance(batching, LengthBatching):
            max_num_elements = batching.max_num_elements

            pinned_memory = device.type == "cuda"
            # Pack.
            builder.pack(
                max_num_elements + 1,
                max_seq_len,
                pad_value=pad_idx,
                truncate=True,
                pinned_memory=pinned_memory,
            )
            BatchLayout.compiled_max_seq_len = max_seq_len
        else:
            raise NotSupportedError(f"`{batching}` is not supported.")

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Convert to `SequenceBatch`.
        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            seqs, seq_lens = example["seqs"], example["seq_lens"]

            return SequenceBatch(seqs, seq_lens, packed=True)

        pipeline = builder.map(to_batch).prefetch(options.num_prefetch).and_return()

        return pipeline
