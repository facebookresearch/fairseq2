# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, NoReturn, cast, final

import torch
from typing_extensions import override

from fairseq2.assets import AssetCard, AssetError
from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.data.text import TextTokenizer
from fairseq2.datasets.batching import Batching, LengthBatching, StaticBatching
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.datasets.utils import _load_files_and_weights
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask


class InstructionDataset(ABC):
    """Represents an instruction finetuning dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        batching: Batching,
        *,
        sample: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        drop_remainder: bool = False,
        sync_batches: bool = True,
        sync_mode: Literal["until_first", "until_last"] = "until_first",
        max_num_batches: int | None = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        src_encode_mode: str = "prompt",
        tgt_encode_mode: str = "prompt_response",
        seed: int = 2,
        **extras: Any,
    ) -> DataReader[SequenceBatch]:
        """Create a dataset reader.

        :param split:
            The split to read.
        :param tokenizer:
            The tokenizer to encode text.
        :param gang:
            The gang over which to shard the dataset.
        :param max_seq_len:
            The maximum sequence length of each example. Examples longer than
            this value will be dropped.
        :param batching:
            The batching strategy for returned examples.
        :param sample:
            If ``True``, instruction sources (e.g. files) will be sampled in
            proportion to their weights.
        :param example_shuffle_window:
            The size of the sliding window for shuffling examples. If ``1``, no
            shuffling is performed; if ``0``, true shuffling is performed by
            loading the entire dataset.
        :param batch_shuffle_window:
            The size of the sliding window for shuffling batches. If ``1``, no
            shuffling is performed; if ``0``, true shuffling is performed by
            loading the entire dataset.
        :param drop_remainder:
            If ``True``, drops the last set of batches if they have in total
            fewer examples than requested.
        :param sync_batches:
            If ``True``, ensures that each process in ``gang`` reads the same
            number of batches. Typically used when the amount of data to be read
            can vary per process (e.g. due to unbalanced sharding or non-static
            batching) and it is critical for each process to iterate over the
            same number of batches (e.g. during training).
        :param sync_mode:
            If ``until_first``, stops iteration when the first rank reaches end
            of data. If ``until_last``, stops iteration when the last rank
            reaches end of data; ranks that have already reached their end of
            data will return an empty list of batches.
        :param max_num_batches:
            The maximum number of batches to return.
        :param num_accumulate:
            The number of batches to accumulate in each iteration. Typically
            used with gradient accumulation during training.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param seed:
            The seed to initialize the random number generators used internally.
        :param src_encode_mode:
            The mode to encode the prompt
        :param tgt_encode_mode:
            The mode to encode the target
        :param extras:
            The extra parameters specific to the dataset implementation.
        """

    @abstractmethod
    def create_prompt_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        batching: StaticBatching,
        *,
        drop_remainder: bool = False,
        sync_batches: bool = True,
        sync_mode: Literal["until_first", "until_last"] = "until_first",
        num_prefetch: int = 1,
        src_encode_mode: str = "prompt",
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        """Create a dataset reader for evaluation.

        :param split:
            The split to read.
        :param tokenizer:
            The tokenizer to encode text.
        :param gang:
            The gang over which to shard the dataset.
        :param max_seq_len:
            The maximum sequence length of each example. Examples longer than
            this value will be dropped.
        :param batching:
            The batching strategy for returned examples.
        :param drop_remainder:
            If ``True``, drops the last batch if it has fewer examples than
            requested.
        :param sync_batches:
            If ``True``, ensures that each process in ``gang`` reads the same
            number of batches. Typically used when the amount of data to be read
            can vary per process (e.g. due to unbalanced sharding or non-static
            batching) and it is critical for each process to iterate over the
            same number of batches (e.g. during training).
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param src_encode_mode:
            The mode to encode the prompt
        :param extras:
            The extra parameters specific to the dataset implementation.
        """

    @abstractmethod
    def splits(self) -> set[str]:
        """Return the set of splits."""


load_instruction_dataset = DelegatingDatasetLoader[InstructionDataset]()


# TODO: FIX, INFER
npc = 10


# TODO: Work in progress!
@final
class GenericInstructionDataset(InstructionDataset):
    """Represents a generic JSONL instruction dataset."""

    _splits: dict[str, tuple[Sequence[Path], Sequence[float]]]

    def __init__(
        self, splits: dict[str, tuple[Sequence[Path], Sequence[float]]]
    ) -> None:
        """
        :param files:
            The instruction files.
        :param weights:
            The weight of each file in ``files``.
        """
        for split, (files, weights) in splits.items():
            if len(files) != len(weights):
                raise ValueError(
                    f"The lengths of the file and weight lists of the '{split}' split must match, but they are {len(files)} and {len(weights)} instead."
                )

        self._splits = splits

    @classmethod
    def from_path(cls, path: Path) -> GenericInstructionDataset:
        """Load a :class:`InstructionDataset` from ``path``."""
        splits: dict[str, tuple[Sequence[Path], Sequence[float]]] = {}

        for child_path in path.iterdir():
            if child_path.is_dir():
                files, weights = _load_files_and_weights(child_path)

                splits[child_path.name] = (files, weights)

        if not splits:
            files, weights = _load_files_and_weights(path)

            splits["default"] = (files, weights)

        return GenericInstructionDataset(splits)

    @override
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        batching: Batching,
        *,
        sample: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        drop_remainder: bool = False,
        sync_batches: bool = True,
        sync_mode: Literal["until_first", "until_last"] = "until_first",
        max_num_batches: int | None = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        src_encode_mode: str = "prompt",
        tgt_encode_mode: str = "prompt_response",
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        try:
            files, weights = self._splits[split]
        except KeyError:
            self._raise_split_error(split)

        if len(files) == 1:
            builder = self._read_jsonl(files[0], tokenizer)
        else:
            pipelines = []

            for file in files:
                pipeline = self._read_jsonl(file, tokenizer).and_return()

                pipelines.append(pipeline)

            if sample:
                builder = DataPipeline.sample(pipelines, weights=weights, seed=seed)

                seed += 1
            else:
                builder = DataPipeline.concat(pipelines)

        # Shuffle files. Must be consistent across all processes.
        if example_shuffle_window != 1:
            builder.shuffle(shuffle_window=0, seed=seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        # Encode prompt and target texts.
        prompt_encoder = tokenizer.create_encoder(mode=src_encode_mode)
        target_encoder = tokenizer.create_encoder(mode=tgt_encode_mode)

        builder.map(prompt_encoder, selector="src", num_parallel_calls=npc)
        builder.map(target_encoder, selector="tgt", num_parallel_calls=npc)

        def cat_source_and_target(example: dict[str, Any]) -> dict[str, Any]:
            id_ = example.get("id")

            prompt_indices = example["src"]
            target_indices = example["tgt"]

            indices = torch.cat([prompt_indices, target_indices])

            target_mask = torch.arange(len(indices)) >= len(prompt_indices)

            return {"id": id_, "indices": indices, "target_mask": target_mask}

        builder.map(cat_source_and_target, num_parallel_calls=npc)

        if isinstance(batching, LengthBatching):
            bucket_sizes = create_bucket_sizes(
                max_seq_len=max_seq_len, max_num_elements=batching.max_num_elements
            )

            # Bucket by the sequence length.
            builder.bucket_by_length(
                bucket_sizes,
                selector="indices",
                skip_above_max_examples=True,
                drop_remainder=drop_remainder,
            )
        elif isinstance(batching, StaticBatching):
            # Filter out long examples.
            def skip(example: dict[str, Any]) -> bool:
                return len(example["indices"]) <= max_seq_len

            builder.filter(skip)

            # Bucket `batch_size` examples.
            builder.bucket(batching.batch_size, drop_remainder=drop_remainder)
        else:
            raise RuntimeError(f"`{batching}` is not supported.")

        # Shuffle buckets.
        if batch_shuffle_window != 1:
            builder.shuffle(batch_shuffle_window, seed=seed)

        seed += 1

        # Collate bucketed examples into a batch.
        target_mask_collate_opts = CollateOptionsOverride(
            "target_mask", pad_value=False
        )

        collater = Collater(pad_value=0, overrides=[target_mask_collate_opts])

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(num_prefetch)

        # Wrap examples with `SequenceBatch`.
        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            indices = cast(SequenceData, example["indices"])

            seqs, padding_mask = get_seqs_and_padding_mask(indices, gang.device)

            target_mask = example["target_mask"]["seqs"].to(gang.device)

            return SequenceBatch(seqs, padding_mask, target_mask, example=example)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=drop_remainder,
            sync_batches=sync_batches,
            sync_mode=sync_mode,
        )

    @override
    def create_prompt_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        batching: StaticBatching,
        *,
        drop_remainder: bool = False,
        sync_batches: bool = True,
        sync_mode: Literal["until_first", "until_last"] = "until_first",
        num_prefetch: int = 1,
        src_encode_mode: str = "prompt",
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        try:
            files, weights = self._splits[split]
        except KeyError:
            self._raise_split_error(split)

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

        # Encode prompt texts.
        text_encoder = tokenizer.create_encoder(mode=src_encode_mode)

        def encode(example: dict[str, Any]) -> dict[str, Any]:
            id_ = example.get("id")

            prompt = example["src"]

            indices = text_encoder(prompt)

            return {"id": id_, "prompt": prompt, "indices": indices}

        builder.map(encode, num_parallel_calls=npc)

        # Filter out long examples.
        def skip(example: dict[str, Any]) -> bool:
            return len(example["indices"]) <= max_seq_len

        builder.filter(skip)

        # Bucket `batch_size` examples.
        builder.bucket(batching.batch_size, drop_remainder=drop_remainder)

        # Collate bucketed examples into a batch.
        collater = Collater(pad_value=tokenizer.vocab_info.pad_idx or 0)

        builder.map(collater, num_parallel_calls=npc)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(num_prefetch)

        # Wrap examples with `SequenceBatch`.
        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            indices = cast(SequenceData, example["indices"])

            seqs, padding_mask = get_seqs_and_padding_mask(indices, gang.device)

            return SequenceBatch(seqs, padding_mask, example=example)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline,
            gang,
            drop_remainder=drop_remainder,
            sync_batches=sync_batches,
            sync_mode=sync_mode,
        )

    def _read_jsonl(self, path: Path, tokenizer: TextTokenizer) -> DataPipelineBuilder:
        lines = []

        # TODO(balioglu): Do in C++.
        with path.open() as fp:
            for line in fp:
                lines.append(line)

        return read_sequence(lines).map(json.loads, num_parallel_calls=npc)

    def _raise_split_error(self, split: str) -> NoReturn:
        raise ValueError(
            f"`split` must be one of the following splits, but is '{split}' instead: {', '.join(sorted(self._splits.keys()))}"
        ) from None

    @override
    def splits(self) -> set[str]:
        return set(self._splits.keys())


@final
class GenericInstructionDatasetLoader(AbstractDatasetLoader[GenericInstructionDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> GenericInstructionDataset:
        try:
            return GenericInstructionDataset.from_path(path)
        except RuntimeError as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex


load_generic_instruction_dataset = GenericInstructionDatasetLoader()

load_instruction_dataset.register(
    "generic_instruction", load_generic_instruction_dataset
)
