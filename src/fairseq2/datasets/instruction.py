# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Dict, Set, cast, final

import torch
from torch import Tensor

from fairseq2.assets import AssetCard
from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    SequenceData,
    create_bucket_sizes,
    list_files,
    read_sequence,
)
from fairseq2.data.text import TextTokenizer
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import override


class InstructionDataset(ABC):
    """Represents an instruction finetuning dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        max_num_tokens: int,
        *,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
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
        :param max_num_tokens:
            The maximum number of tokens in each batch.
        :param example_shuffle_window:
            The size of the sliding window for shuffling examples. If ``1``, no
            shuffling is performed; if ``0``, true shuffling is performed by
            loading the entire dataset.
        :param batch_shuffle_window:
            The size of the sliding window for shuffling batches. If ``1``, no
            shuffling is performed; if ``0``, true shuffling is performed by
            loading the entire dataset.
        :param num_accumulate:
            The number of batches to accumulate in each iteration. Typically
            used with gradient accumulation during training.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param seed:
            The seed to initialize the random number generators used internally.
        :param extras:
            The extra parameters specific to the dataset implementation.
        """

    @abstractmethod
    def create_eval_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        *,
        batch_size: int = 1,
        num_prefetch: int = 1,
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        """Create a dataset reader for evaluation.

        :param split:
            The split to read.
        :param tokenizer:
            The tokenizer to encode text.
        :param gang:
            The gang over which to shard the dataset.
        :param batch_size:
            The size of returned batches.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param extras:
            The extra parameters specific to the dataset implementation.
        """

    @abstractmethod
    def splits(self) -> Set[str]:
        """Return the set of splits."""


load_instruction_dataset = DelegatingDatasetLoader[InstructionDataset]()


# TODO: FIX, INFER
npc = 10


# TODO: Work in progress!
@final
class GenericInstructionDataset(InstructionDataset):
    """Represents a generic JSONL instruction dataset."""

    _data_dir: Path

    def __init__(self, data_dir: Path) -> None:
        """
        :param data_dir:
            The directory under which the JSONL files reside.
        """
        self._data_dir = data_dir

    @override
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        max_num_tokens: int,
        *,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        builder = list_files(self._data_dir, pattern="*.jsonl")

        # Shuffle files. Must be consistent across all processes.
        if example_shuffle_window != 1:
            builder.shuffle(shuffle_window=0, seed=seed)

        seed += 1

        builder.yield_from(partial(self._read_jsonl, tokenizer=tokenizer))

        # Shuffle examples.
        if example_shuffle_window != 1:
            builder.shuffle(example_shuffle_window, seed=seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        bucket_sizes = create_bucket_sizes(
            max_num_elements=max_num_tokens, max_seq_len=max_seq_len
        )

        # Bucket by token sequence length.
        builder.bucket_by_length(
            bucket_sizes, selector="tokens", skip_above_max_examples=True
        )

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

        # Prefetch `num_prefetch` examples in background.
        builder.prefetch(num_prefetch)

        # Wrap examples with `SequenceBatch`.
        def example_to_batch(example: Dict[str, Any]) -> SequenceBatch:
            text = cast(SequenceData, example["tokens"])

            seqs, padding_mask = get_seqs_and_padding_mask(text, gang.device)

            target_mask = example["target_mask"]["seqs"].to(gang.device)

            return SequenceBatch(seqs, padding_mask, target_mask, example=example)

        pipeline = builder.map(example_to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=True,
        )

    # TODO: cache
    def _read_jsonl(self, path: str, tokenizer: TextTokenizer) -> DataPipeline:
        lines = []

        with Path(path).open() as fp:
            for line in fp:
                lines.append(line)

        builder = read_sequence(lines)

        builder.map(json.loads, num_parallel_calls=npc)

        source_text_encoder = tokenizer.create_encoder(mode="prompt")
        target_text_encoder = tokenizer.create_encoder(mode="prompt_response")

        builder.map(source_text_encoder, selector="src", num_parallel_calls=npc)
        builder.map(target_text_encoder, selector="tgt", num_parallel_calls=npc)

        def cat_source_and_text(d: Dict[str, Any]) -> Dict[str, Tensor]:
            source_tokens = d["src"]
            target_tokens = d["tgt"]

            tokens = torch.cat([source_tokens, target_tokens])

            target_mask = torch.arange(len(tokens)) >= len(source_tokens)

            return {"tokens": tokens, "target_mask": target_mask}

            builder.map(cat_source_and_text, num_parallel_calls=npc)

        return builder.and_return()

    @override
    def create_eval_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        *,
        batch_size: int = 1,
        num_prefetch: int = 1,
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        builder = list_files(self._data_dir, pattern="*.jsonl")

        text_encoder = tokenizer.create_encoder(mode="prompt")

        def read_jsonl(path: str) -> DataPipeline:
            lines = []

            with Path(path).open() as fp:
                for line in fp:
                    lines.append(line)

            builder = read_sequence(lines)

            builder.map(json.loads, num_parallel_calls=npc)

            builder.map(lambda e: {"prompt": e["src"], "tokens": e["src"]})

            builder.map(text_encoder, selector="tokens", num_parallel_calls=npc)

            return builder.and_return()

        builder.yield_from(read_jsonl)

        builder.shard(gang.rank, gang.size)

        builder.bucket(batch_size)

        collater = Collater(pad_value=0)

        builder.map(collater, num_parallel_calls=npc)

        builder.prefetch(num_prefetch)

        def _example_to_batch(example: Dict[str, Any]) -> SequenceBatch:
            text = cast(SequenceData, example["tokens"])

            seqs, padding_mask = get_seqs_and_padding_mask(text, gang.device)

            return SequenceBatch(seqs, padding_mask, example=example)

        pipeline = builder.map(_example_to_batch).and_return()

        return DataPipelineReader[SequenceBatch](pipeline, gang)

    @override
    def splits(self) -> Set[str]:
        return {"train"}


@final
class GenericInstructionDatasetLoader(AbstractDatasetLoader[GenericInstructionDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> GenericInstructionDataset:
        return GenericInstructionDataset(path)


load_generic_instruction_dataset = GenericInstructionDatasetLoader()

load_instruction_dataset.register(
    "generic_instruction", load_generic_instruction_dataset
)
