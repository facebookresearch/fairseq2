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
from typing import Any, cast, final

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
from fairseq2.datasets.data_reader import DataPipelineReader
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.datasets.utils import _load_files_and_weights
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask


@dataclass
class PreferenceOptimizationBatch:
    """Represents a preference optimization dataset batch."""

    chosen: SequenceBatch
    rejected: SequenceBatch


class PreferenceOptimizationDataset(ABC):
    """Represents a preference optimization dataset."""

    @abstractmethod
    def create_reader(
        self,
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
        max_num_batches: int | None = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        mask_source_tokens: bool = True,
        src_encode_mode: str = "prompt",
        tgt_encode_mode: str = "prompt_response",
        seed: int = 2,
        **extras: Any,
    ) -> DataPipelineReader[PreferenceOptimizationBatch]:
        """Create a dataset reader.

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
        :param max_num_batches:
            The maximum number of batches to return.
        :param num_accumulate:
            The number of batches to accumulate in each iteration. Typically
            used with gradient accumulation during training.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param mask_source_tokens:
            If ``False``, calculates loss on the `src` tokens as well as the `tgt` tokens.
        :param src_encode_mode:
            The mode to encode the prompt
        :param tgt_encode_mode:
            The mode to encode the target
        :param seed:
            The seed to initialize the random number generators used internally.
        :param extras:
            The extra parameters specific to the dataset implementation.
        """


load_preference_optimization_dataset = DelegatingDatasetLoader[
    PreferenceOptimizationDataset
]()

# TODO: FIX, INFER
npc = 10


@final
class GenericPreferenceOptimizationDataset(PreferenceOptimizationDataset):
    """Represents a generic JSONL preference optimization dataset."""

    _files: Sequence[Path]
    _weights: Sequence[float]

    def __init__(self, files: Sequence[Path], weights: Sequence[float]) -> None:
        """
        :param files:
            The instruction files.
        :param weights:
            The weight of each file in ``files``.
        """
        if len(files) != len(weights):
            raise ValueError(
                f"The lengths of `files` and `weights` must match, but they are {len(files)} and {len(weights)} instead."
            )

        self._files = files
        self._weights = weights

    @classmethod
    def from_path(cls, path: Path) -> GenericPreferenceOptimizationDataset:
        """Load a :class:`PreferenceOptimizationDataset` from ``path``."""
        files, weights = _load_files_and_weights(path)

        return GenericPreferenceOptimizationDataset(files, weights)

    @override
    def create_reader(
        self,
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
        max_num_batches: int | None = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        mask_source_tokens: bool = True,
        src_encode_mode: str = "prompt",
        tgt_encode_mode: str = "prompt_response",
        seed: int = 2,
        **extras: Any,
    ) -> DataPipelineReader[PreferenceOptimizationBatch]:
        if len(self._files) == 1:
            builder = self._read_jsonl(self._files[0], tokenizer)
        else:
            pipelines = []

            for file in self._files:
                pipeline = self._read_jsonl(file, tokenizer).and_return()

                pipelines.append(pipeline)

            if sample:
                builder = DataPipeline.sample(
                    pipelines, weights=self._weights, seed=seed
                )

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
        builder.map(target_encoder, selector="tgt_chosen", num_parallel_calls=npc)
        builder.map(target_encoder, selector="tgt_rejected", num_parallel_calls=npc)

        def cat_source_and_target(example: dict[str, Any]) -> dict[str, Any]:
            id_ = example.get("id", None)

            prompt_indices = example["src"]
            target_indices_chosen = example["tgt_chosen"]
            target_indices_rejected = example["tgt_rejected"]

            indices_chosen = torch.cat([prompt_indices, target_indices_chosen])
            indices_rejected = torch.cat([prompt_indices, target_indices_rejected])

            if mask_source_tokens:
                prompt_len = len(prompt_indices)
                target_mask_chosen = torch.arange(len(indices_chosen)) >= prompt_len
                target_mask_rejected = torch.arange(len(indices_rejected)) >= prompt_len
            else:
                target_mask_chosen = torch.full([len(indices_chosen)], True)
                target_mask_rejected = torch.full([len(indices_rejected)], True)

            total_tokens = (
                2 * len(prompt_indices)
                + len(target_indices_chosen)
                + len(target_indices_rejected)
            )

            return {
                "id": id_,
                "indices_prompt": prompt_indices,
                "indices_chosen": indices_chosen,
                "indices_rejected": indices_rejected,
                "target_mask_chosen": target_mask_chosen,
                "target_mask_rejected": target_mask_rejected,
                "total_tokens": total_tokens,
            }

        builder.map(cat_source_and_target, num_parallel_calls=npc)

        if isinstance(batching, LengthBatching):
            bucket_sizes = create_bucket_sizes(
                max_seq_len=max_seq_len, max_num_elements=batching.max_num_elements
            )

            # Bucket by the sequence length.
            builder.bucket_by_length(
                bucket_sizes,
                selector="total_tokens",
                skip_above_max_examples=True,
                drop_remainder=drop_remainder,
            )
        elif isinstance(batching, StaticBatching):
            # Filter out long examples.
            def skip(example: dict[str, Any]) -> bool:
                chosen_len = len(example["indices_chosen"])
                rejected_len = len(example["indices_rejected"])

                return chosen_len <= max_seq_len and rejected_len <= max_seq_len

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
        target_mask_collate_opts = [
            CollateOptionsOverride("target_mask_chosen", pad_value=False),
            CollateOptionsOverride("target_mask_rejected", pad_value=False),
        ]

        collater = Collater(pad_value=0, overrides=target_mask_collate_opts)

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(num_prefetch)

        # Wrap examples with `PreferenceOptimizationBatch`.
        def to_batch(example: dict[str, Any]) -> PreferenceOptimizationBatch:
            indices_chosen = cast(SequenceData, example["indices_chosen"])
            indices_rejected = cast(SequenceData, example["indices_rejected"])

            seqs_chosen, padding_mask_chosen = get_seqs_and_padding_mask(
                indices_chosen, gang.device
            )
            seqs_rejected, padding_mask_rejected = get_seqs_and_padding_mask(
                indices_rejected, gang.device
            )

            target_mask_chosen = example["target_mask_chosen"]["seqs"].to(gang.device)
            target_mask_rejected = example["target_mask_rejected"]["seqs"].to(gang.device)  # fmt: skip

            batch_chosen = SequenceBatch(
                seqs_chosen,
                padding_mask_chosen,
                target_mask_chosen,
                example=example,
            )

            batch_rejected = SequenceBatch(
                seqs_rejected,
                padding_mask_rejected,
                target_mask_rejected,
                example=example,
            )

            return PreferenceOptimizationBatch(batch_chosen, batch_rejected)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[PreferenceOptimizationBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=drop_remainder,
            sync_batches=sync_batches,
        )

    def _read_jsonl(self, path: Path, tokenizer: TextTokenizer) -> DataPipelineBuilder:
        lines = []

        # TODO(balioglu): Do in C++.
        with path.open() as fp:
            for line in fp:
                lines.append(line)

        return read_sequence(lines).map(json.loads, num_parallel_calls=npc)


@final
class GenericPreferenceOptimizationDatasetLoader(
    AbstractDatasetLoader[GenericPreferenceOptimizationDataset]
):
    @override
    def _load(
        self, path: Path, card: AssetCard
    ) -> GenericPreferenceOptimizationDataset:
        try:
            return GenericPreferenceOptimizationDataset.from_path(path)
        except RuntimeError as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex


load_generic_preference_optimization_dataset = (
    GenericPreferenceOptimizationDatasetLoader()
)

load_preference_optimization_dataset.register(
    "generic_preference_optimization", load_generic_preference_optimization_dataset
)
