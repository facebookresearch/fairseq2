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
from fairseq2.datasets import (
    DataPipelineReader,
    DataReadOptions,
    DatasetHubAccessor,
    LengthBatching,
    StaticBatching,
)
from fairseq2.datasets._utils import _load_files_and_weights
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask


@dataclass(kw_only=True)
class PreferenceReadOptions(DataReadOptions):
    sample: bool = False
    """
    If ``True``, instruction sources (e.g. JSONL files) will be sampled in
    proportion to their weights.
    """

    mask_source_tokens: bool = True
    """
    If ``False``, calculates loss on the source tokens (prompt) as well as the
    target tokens.
    """

    source_encode_mode: str = "prompt"
    """The tokenizer mode to encode the source text."""

    target_encode_mode: str = "prompt_response"
    """The tokenizer mode to encode the target text."""


@dataclass
class PreferenceBatch:
    """Represents a preference optimization dataset batch."""

    chosen: SequenceBatch
    rejected: SequenceBatch
    reference_score_chosen: torch.Tensor | None
    reference_score_rejected: torch.Tensor | None


class PreferenceDataset(ABC):
    """Represents a preference optimization dataset."""

    @abstractmethod
    def create_reader(
        self,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: PreferenceReadOptions | None = None,
    ) -> DataPipelineReader[PreferenceBatch]:
        """Create a dataset reader.

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


# TODO: FIX, INFER
npc = 10


GENERIC_PREFERENCE_DATASET_FAMILY: Final = "generic_preference"


@final
class GenericPreferenceDataset(PreferenceDataset):
    """Represents a generic JSONL preference optimization dataset."""

    _name: str
    _files: Sequence[Path]
    _weights: Sequence[float]

    def __init__(
        self, name: str, files: Sequence[Path], weights: Sequence[float]
    ) -> None:
        """
        :param files:
            The instruction files.
        :param weights:
            The weight of each file in ``files``.
        """
        self._name = name

        if len(files) != len(weights):
            raise ValueError(
                f"The lengths of `files` and `weights` must match, but they are {len(files)} and {len(weights)} instead."
            )

        self._files = files
        self._weights = weights

    @staticmethod
    def from_path(path: Path, name: str) -> GenericPreferenceDataset:
        files, weights = _load_files_and_weights(name, path)

        return GenericPreferenceDataset(name, files, weights)

    @override
    def create_reader(
        self,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: PreferenceReadOptions | None = None,
    ) -> DataPipelineReader[PreferenceBatch]:
        if options is None:
            options = PreferenceReadOptions()

        seed = options.seed

        if len(self._files) == 1:
            builder = self._read_jsonl(self._files[0], tokenizer)
        else:
            pipelines = []

            for file in self._files:
                pipeline = self._read_jsonl(file, tokenizer).and_return()

                pipelines.append(pipeline)

            if options.sample:
                builder = DataPipeline.sample(
                    pipelines, weights=self._weights, seed=seed
                )

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

        # Encode source and target texts.
        source_encoder = tokenizer.create_encoder(mode=options.source_encode_mode)
        target_encoder = tokenizer.create_encoder(mode=options.target_encode_mode)

        builder.map(source_encoder, selector="src", num_parallel_calls=npc)
        builder.map(target_encoder, selector="tgt_chosen", num_parallel_calls=npc)
        builder.map(target_encoder, selector="tgt_rejected", num_parallel_calls=npc)

        def cat_source_and_target(example: dict[str, Any]) -> dict[str, Any]:
            id_ = example.get("id", None)

            source_indices = example["src"]
            target_indices_chosen = example["tgt_chosen"]
            target_indices_rejected = example["tgt_rejected"]

            indices_chosen = torch.cat([source_indices, target_indices_chosen])
            indices_rejected = torch.cat([source_indices, target_indices_rejected])

            if options.mask_source_tokens:
                source_len = len(source_indices)
                target_mask_chosen = torch.arange(len(indices_chosen)) >= source_len
                target_mask_rejected = torch.arange(len(indices_rejected)) >= source_len
            else:
                target_mask_chosen = torch.full([len(indices_chosen)], True)
                target_mask_rejected = torch.full([len(indices_rejected)], True)

            total_tokens = (
                2 * len(source_indices)
                + len(target_indices_chosen)
                + len(target_indices_rejected)
            )

            # below is an example of using extras field of data reader options
            if "keep_jsonl_keys" in options.extras:
                jsonl_keys = options.extras["keep_jsonl_keys"]
                if not (
                    isinstance(jsonl_keys, list)
                    and all(isinstance(i, str) for i in jsonl_keys)
                ):
                    raise ValueError(f"{jsonl_keys} must be a list of strings")
                jsonl_content = {k: example.get(k, None) for k in jsonl_keys}
            else:
                jsonl_content = None

            return {
                "id": id_,
                "indices_prompt": source_indices,
                "indices_chosen": indices_chosen,
                "indices_rejected": indices_rejected,
                "reference_score_chosen": example.get("reference_score_chosen", None),
                "reference_score_rejected": example.get(
                    "reference_score_rejected", None
                ),
                "target_mask_chosen": target_mask_chosen,
                "target_mask_rejected": target_mask_rejected,
                "total_tokens": total_tokens,
                "keep_jsonl_keys": jsonl_content,
            }

        builder.map(cat_source_and_target, num_parallel_calls=npc)

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
                selector="total_tokens",
                min_data_len=min_seq_len,
                skip_above_max_examples=True,
                drop_remainder=options.drop_remainder,
            )
        elif isinstance(batching, StaticBatching):
            # Filter out long examples.
            def skip(example: dict[str, Any]) -> bool:
                chosen_len = len(example["indices_chosen"])
                rejected_len = len(example["indices_rejected"])

                if chosen_len > max_seq_len or rejected_len > max_seq_len:
                    return False

                return chosen_len >= min_seq_len and rejected_len >= min_seq_len

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
        target_mask_collate_opts = [
            CollateOptionsOverride("target_mask_chosen", pad_value=False),
            CollateOptionsOverride("target_mask_rejected", pad_value=False),
        ]

        collater = Collater(pad_value=0, overrides=target_mask_collate_opts)

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(options.num_prefetch)

        # Wrap examples with `PreferenceBatch`.
        def to_batch(example: dict[str, Any]) -> PreferenceBatch:
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

            batch_reference_scores_chosen = None
            if all(example["reference_score_chosen"]):
                batch_reference_scores_chosen = torch.Tensor(
                    example["reference_score_chosen"]
                ).to(gang.device)
            batch_reference_scores_rejected = None
            if all(example["reference_score_rejected"]):
                batch_reference_scores_rejected = torch.Tensor(
                    example["reference_score_rejected"]
                ).to(gang.device)

            return PreferenceBatch(
                batch_chosen,
                batch_rejected,
                batch_reference_scores_chosen,
                batch_reference_scores_rejected,
            )

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[PreferenceBatch](
            self._name, "default", pipeline, gang, options
        )

    def _read_jsonl(self, path: Path, tokenizer: TextTokenizer) -> DataPipelineBuilder:
        lines = []

        # TODO(balioglu): Do in C++.
        with path.open() as fp:
            for line in fp:
                lines.append(line)

        return read_sequence(lines).map(json.loads, num_parallel_calls=npc)


get_preference_dataset_hub = DatasetHubAccessor(PreferenceDataset)
