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
from typing import Any, Final, cast, final

from fairseq2.data.data_pipeline import (
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    read_sequence,
)
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import (
    DataPipelineReader,
    DataReadOptions,
    SequenceBatch,
    StaticBatching,
    UnknownSplitError,
)
from fairseq2.datasets._utils import _load_files_and_weights
from fairseq2.error import InfraError, NotSupportedError
from fairseq2.gang import Gang
from fairseq2.runtime.dependency import DependencyResolver


@dataclass(kw_only=True)
class InstructionReadOptions(DataReadOptions):
    sample: bool = False
    """
    If ``True``, instruction sources (e.g. JSONL files) will be sampled in
    proportion to their weights.
    """

    source_encode_mode: str = "prompt"
    """The tokenizer mode to encode the source text."""

    target_encode_mode: str = "prompt_response"
    """The tokenizer mode to encode the target text."""


@dataclass
class InstructionPromptReadOptions(DataReadOptions):
    source_encode_mode: str = "prompt"
    """The tokenizer mode to encode the source text."""


# TODO: FIX, INFER
npc = 10


INSTRUCTION_DATASET_FAMILY: Final = "generic_instruction"


# TODO: Work in progress!
@final
class InstructionDataset:
    """Represents a generic JSONL instruction dataset."""

    _name: str
    _splits: dict[str, tuple[Sequence[Path], Sequence[float]]]

    def __init__(
        self, name: str, splits: dict[str, tuple[Sequence[Path], Sequence[float]]]
    ) -> None:
        """
        :param files:
            The instruction files.
        :param weights:
            The weight of each file in ``files``.
        """
        self._name = name

        for split, (files, weights) in splits.items():
            if len(files) != len(weights):
                raise ValueError(
                    f"The lengths of the file and weight lists of the '{split}' split must match, but they are {len(files)} and {len(weights)} instead."
                )

        self._splits = splits

    @staticmethod
    def from_path(
        resolver: DependencyResolver, path: Path, name: str
    ) -> InstructionDataset:
        splits: dict[str, tuple[Sequence[Path], Sequence[float]]] = {}

        if path.is_dir():
            try:
                child_dirs = [p for p in path.iterdir() if p.is_dir()]
            except OSError as ex:
                raise InfraError(
                    f"The files under the '{path}' directory of the '{name}' dataset cannot be retrieved. See the nested exception for details."  # fmt: skip
                ) from ex

            for child_dir in child_dirs:
                files, weights = _load_files_and_weights(name, child_dir)

                splits[child_dir.name] = (files, weights)

        if not splits:
            files, weights = _load_files_and_weights(name, path)

            splits["default"] = (files, weights)

        return InstructionDataset(name, splits)

    def create_prompt_reader(
        self,
        split: str,
        tokenizer: Tokenizer,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: InstructionPromptReadOptions | None = None,
    ) -> DataPipelineReader[SequenceBatch]:
        try:
            files, weights = self._splits[split]
        except KeyError:
            raise UnknownSplitError(self._name, split, self._splits.keys()) from None

        if options is None:
            options = InstructionPromptReadOptions()

        if len(files) == 1:
            builder = self._read_jsonl(files[0], tokenizer)
        else:
            pipelines = []

            for file in files:
                pipeline = self._read_jsonl(file, tokenizer).and_return()

                pipelines.append(pipeline)

            builder = DataPipeline.concat(pipelines)

        # Shard
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        # Encode source texts.
        text_encoder = tokenizer.create_encoder(mode=options.source_encode_mode)

        def encode(example: dict[str, Any]) -> dict[str, Any]:
            id_ = example.get("id")

            source = example["src"]

            indices = text_encoder(source)

            return {"id": id_, "prompt": source, "indices": indices}

        builder.map(encode, num_parallel_calls=npc)

        # Filter out long examples.
        def skip(example: dict[str, Any]) -> bool:
            seq_len = len(example["indices"])

            return seq_len >= min_seq_len and seq_len <= max_seq_len

        builder.filter(skip)

        batching = options.batching

        if not isinstance(batching, StaticBatching):
            raise NotSupportedError(f"`{batching}` is not supported.")

        # Bucket `batch_size` examples.
        builder.bucket(batching.batch_size, drop_remainder=options.drop_remainder)

        # Collate bucketed examples into a batch.
        collater = Collater(pad_value=tokenizer.vocab_info.pad_idx or 0)

        builder.map(collater, num_parallel_calls=npc)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(options.num_prefetch)

        # Wrap examples with `SequenceBatch`.
        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            indices = cast(SequenceData, example["indices"])

            seqs, seq_lens = indices["seqs"], indices["seq_lens"]

            return SequenceBatch(seqs, seq_lens, example=example)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            self._name, split, pipeline, gang, options
        )

    def _read_jsonl(self, path: Path, tokenizer: Tokenizer) -> DataPipelineBuilder:
        lines = []

        # TODO(balioglu): Do in C++.
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                lines.append(line)

        return read_sequence(lines).map(json.loads, num_parallel_calls=npc)

    def splits(self) -> set[str]:
        return set(self._splits.keys())
