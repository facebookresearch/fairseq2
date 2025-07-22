# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast, final

import torch
from typing_extensions import override

from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.data.text.tokenizers.hg import HuggingFaceTokenEncoder
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DataReadOptions,
    DatasetHubAccessor,
    DatasetLoadError,
    LengthBatching,
    SequenceBatch,
    StaticBatching,
    UnknownSplitError,
)
from fairseq2.datasets.utils._manifest import _load_files_and_weights
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang


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

    chat_mode: bool = False


@dataclass
class InstructionPromptReadOptions(DataReadOptions):
    source_encode_mode: str = "prompt"
    """The tokenizer mode to encode the source text."""


class InstructionDataset(ABC):
    """Represents an instruction finetuning dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: InstructionReadOptions | None = None,
    ) -> DataReader[SequenceBatch]:
        """Create a dataset reader.

        :param split:
            The split to read.
        :param tokenizer:
            The tokenizer to encode text.
        :param gang:
            The gang over which to shard the dataset.
        :param min_seq_len:
            The minimum sequence length of each example. Examples shorter than
            this value will be dropped.
        :param max_seq_len:
            The maximum sequence length of each example. Examples longer than
            this value will be dropped.
        :param options:
            The read options.
        """

    @abstractmethod
    def create_prompt_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: InstructionPromptReadOptions | None = None,
    ) -> DataPipelineReader[SequenceBatch]:
        """Create a dataset reader for evaluation.

        :param split:
            The split to read.
        :param tokenizer:
            The tokenizer to encode text.
        :param gang:
            The gang over which to shard the dataset.
        :param min_seq_len:
            The minimum sequence length of each example. Examples shorter than
            this value will be dropped.
        :param max_seq_len:
            The maximum sequence length of each example. Examples longer than
            this value will be dropped.
        :param options:
            The read options.
        """

    @abstractmethod
    def splits(self) -> set[str]:
        """Return the set of splits."""


GENERIC_INSTRUCTION_DATASET_FAMILY: Final = "generic_instruction"


# TODO: Work in progress!
@final
class GenericInstructionDataset(InstructionDataset):
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
    def from_path(path: Path, name: str) -> GenericInstructionDataset:
        splits: dict[str, tuple[Sequence[Path], Sequence[float]]] = {}

        if path.is_dir():
            try:
                child_dirs = [p for p in path.iterdir() if p.is_dir()]
            except OSError as ex:
                raise DatasetLoadError(
                    name, f"The files under the '{path}' directory of the '{name}' dataset cannot be retrieved. See the nested exception for details."  # fmt: skip
                ) from ex

            for child_dir in child_dirs:
                files, weights = _load_files_and_weights(name, child_dir)

                splits[child_dir.name] = (files, weights)

        if not splits:
            files, weights = _load_files_and_weights(name, path)

            splits["default"] = (files, weights)

        return GenericInstructionDataset(name, splits)

    @override
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: InstructionReadOptions | None = None,
    ) -> DataPipelineReader[SequenceBatch]:
        files_weights = self._splits.get(split)
        if files_weights is None:
            raise UnknownSplitError(self._name, split, self._splits.keys())

        if options is None:
            options = InstructionReadOptions()

        seed = options.seed

        files, weights = files_weights

        if len(files) == 1:
            builder = self._read_jsonl(files[0], tokenizer)
        else:
            pipelines = []

            for file in files:
                pipeline = self._read_jsonl(file, tokenizer).and_return()

                pipelines.append(pipeline)

            if options.sample:
                builder = DataPipeline.sample(pipelines, weights=weights, seed=seed)

                seed += 1
            else:
                builder = DataPipeline.concat(pipelines)

        # Shuffle files. Must be consistent across all processes.
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed=seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        if options.chat_mode is True:
            # not passing any encoding modes here, because we use apply_chat_template here
            encoder = tokenizer.create_encoder()
            if not isinstance(encoder, HuggingFaceTokenEncoder):
                raise RuntimeError(
                    "Huggingface tokenizer must be used when chat_mode is True"
                )
            else:

                def encoding_chat(example: dict[str, Any]) -> dict[str, Any]:
                    id_ = example.get("id")
                    chat = example.get("chat")

                    encoded_output = encoder.apply_chat_template(
                        chat,
                        return_dict=True,
                        return_assistant_tokens_mask=True,
                        return_tensors="pt",
                    )

                    indices = encoded_output["input_ids"][0]
                    target_mask = encoded_output["assistant_masks"][0].bool()

                    return {"id": id_, "indices": indices, "target_mask": target_mask}

                builder.map(encoding_chat)

        else:

            # Encode source and target texts.
            source_encoder = tokenizer.create_encoder(mode=options.source_encode_mode)
            target_encoder = tokenizer.create_encoder(mode=options.target_encode_mode)

            builder.map(source_encoder, selector="src")
            builder.map(target_encoder, selector="tgt")

            def cat_source_and_target(example: dict[str, Any]) -> dict[str, Any]:
                id_ = example.get("id")

                source_indices = example["src"]
                target_indices = example["tgt"]

                indices = torch.cat([source_indices, target_indices])

                target_mask = torch.arange(len(indices)) >= len(source_indices)

                return {"id": id_, "indices": indices, "target_mask": target_mask}

            builder.map(cat_source_and_target)

        batching = options.batching

        if isinstance(batching, LengthBatching):
            bucket_sizes = create_bucket_sizes(
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                max_num_elements=batching.max_num_elements,
            )

            # Bucket by the sequence length.
            builder.bucket_by_length(
                bucket_sizes,
                selector="indices",
                min_data_len=min_seq_len,
                skip_above_max_examples=True,
                drop_remainder=options.drop_remainder,
            )
        elif isinstance(batching, StaticBatching):
            # Filter out long examples.
            def skip(example: dict[str, Any]) -> bool:
                seq_len = len(example["indices"])

                return seq_len >= min_seq_len and seq_len <= max_seq_len

            builder.filter(skip)

            # Bucket `batch_size` examples.
            builder.bucket(batching.batch_size, drop_remainder=options.drop_remainder)
        else:
            raise NotSupportedError(f"`{batching}` is not supported.")

        # Shuffle buckets.
        if options.batch_shuffle_window != 1:
            builder.shuffle(options.batch_shuffle_window, seed=seed)

        seed += 1

        # Collate bucketed examples into a batch.
        target_mask_collate_opts = CollateOptionsOverride(
            "target_mask", pad_value=False
        )

        collater = Collater(
            pad_value=tokenizer.vocab_info.pad_idx, overrides=[target_mask_collate_opts]
        )

        builder.map(collater, num_parallel_calls=options.npc)

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

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

    @override
    def create_prompt_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
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

        builder.map(encode)

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

        builder.map(collater, num_parallel_calls=options.npc)

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

    def _read_jsonl(self, path: Path, tokenizer: TextTokenizer) -> DataPipelineBuilder:
        lines = []

        # TODO(balioglu): Do in C++.
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                lines.append(line)

        return read_sequence(lines).map(json.loads)

    @override
    def splits(self) -> set[str]:
        return set(self._splits.keys())


get_instruction_dataset_hub = DatasetHubAccessor(InstructionDataset)
