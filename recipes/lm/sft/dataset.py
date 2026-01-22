# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import torch

from fairseq2.assets import get_asset_download_manager
from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    SequenceData,
    create_bucket_sizes,
)
from fairseq2.data.data_pipeline import DataPipeline, read_sequence
from fairseq2.data.text import read_text
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.data.tokenizers.hg import HuggingFaceTokenEncoder
from fairseq2.datasets import DataPipelineReader, SequenceBatch, SyncMode
from fairseq2.error import NotSupportedError, raise_operational_system_error
from fairseq2.gang import Gangs
from fairseq2.utils.uri import Uri

LM_SFT_DATASET: Final = "lm_sft"


@dataclass
class StaticBatching:
    """Specifies batching where each batch has the same number of examples."""

    batch_size: int
    """The number of examples in each batch."""


@dataclass
class LengthBatching:
    """Specifies batching where each batch has a maximum number of elements."""

    max_num_elements: int
    """The maximum number of elements (e.g. tokens) in each batch."""


Batching: TypeAlias = StaticBatching | LengthBatching


@dataclass(kw_only=True)
class DataReadOptions:
    batching: Batching = field(default_factory=lambda: StaticBatching(1))
    """The batching strategy for returned examples."""

    example_shuffle_window: int = 0
    """
    The size of the sliding window for shuffling examples. If ``1``, no
    shuffling is performed; if ``0``, true shuffling is performed by loading the
    entire dataset.
    """

    batch_shuffle_window: int = 0
    """
    The size of the sliding window for shuffling batches. If ``1``, no
    shuffling is performed; if ``0``, true shuffling is performed by loading the
    entire dataset.
    """

    drop_remainder: bool = False
    """
    If ``True``, drops the last set of batches if they have in total fewer
    examples than requested.
    """

    sync_batches: bool = True
    """
    If ``True``, ensures that each process in the gang reads the same number of
    batches. Typically used when the amount of data to be read can vary per
    process (e.g. due to unbalanced sharding or non-static batching) and it is
    critical for each process to iterate over the same number of batches (e.g.
    during training).
    """

    sync_mode: SyncMode = SyncMode.UNTIL_FIRST
    """
    The data synchronization mode among processes in the gang. Only effective if
    :attr:`sync_batches` is ``True``.
    """

    max_num_batches: int | None = None
    """The maximum number of batches to return."""

    num_accumulate: int = 1
    """
    The number of batches to accumulate in each iteration. Typically used with
    gradient accumulation during training.
    """

    prefetch: int = 1
    """The number of batches to prefetch in background."""

    npc: int = 10
    """The reference number of parallel calls that data reader can do."""

    seed: int = 2
    """The seed to initialize the random number generators used internally."""

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


class LMSFTDataset:
    def __init__(self, sources: dict[str, list[LMSFTDataSource]]) -> None:
        self._sources = sources

    def _create_path_reader(
        self, path: str, split: str | None, gangs: Gangs, shuffle_window: int, seed: int
    ) -> DataPipeline:
        download_manager = get_asset_download_manager()

        uri = Uri.maybe_parse(path)
        if uri:
            local_path = download_manager.download_dataset(uri)
        else:
            local_path = Path(path)

        if split:
            local_path = local_path.joinpath(split)

        if not local_path.is_dir():
            files = [local_path]
        else:
            try:
                files = [f for f in local_path.glob("**/*.jsonl") if not f.is_dir()]
            except OSError as ex:
                raise_operational_system_error(ex)

            files.sort()

        builder = read_sequence(files)

        def read_file(file: Path) -> DataPipeline:
            return read_text(file).map(json.loads).and_return()

        builder.yield_from(read_file)

        if shuffle_window != 1:
            builder.shuffle(shuffle_window, seed=seed)

        builder.shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)

        return builder.and_return()

    def create_reader(
        self,
        split: str,
        tokenizer: Tokenizer,
        gangs: Gangs,
        min_seq_len: int,
        max_seq_len: int,
        options: DataReadOptions | None = None,
    ) -> DataPipelineReader[SequenceBatch]:
        if options is None:
            options = DataReadOptions()

        sources = self._sources[split]

        seed = options.seed

        pipelines = []

        weights = []

        for source in sources:
            pipeline = self._create_path_reader(
                source.path, source.split, gangs, options.example_shuffle_window, seed
            )

            seed += 1

            pipelines.append(pipeline)

            weights.append(source.weight)

        seed += gangs.dp.rank

        builder = DataPipeline.sample(pipelines, weights, seed)

        if options.chat_mode is True:
            # not passing any encoding modes here, because we use apply_chat_template here
            encoder = tokenizer.create_encoder()
            if not isinstance(encoder, HuggingFaceTokenEncoder):
                raise RuntimeError(
                    "Huggingface tokenizer must be used when chat_mode is True"
                )
            else:

                def encoding_chat(example: dict[str, Any]) -> dict[str, Any]:
                    id_ = example.get("id", None)
                    chat = example.get("chat", None)

                    if not chat:
                        chat = [
                            {"role": "user", "content": example.get("src")},
                            {"role": "assistant", "content": example.get("tgt")},
                        ]

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

        if tokenizer.vocab_info.pad_idx is None:
            raise RuntimeError(
                "LMSFTDataset requires pad token to work for batching purposes, check your tokenizer config."
            )

        collater = Collater(
            pad_value=tokenizer.vocab_info.pad_idx, overrides=[target_mask_collate_opts]
        )

        builder.map(collater, num_parallel_calls=options.npc)

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Prefetch `prefetch` batches in background.
        builder.prefetch(options.prefetch)

        # Wrap examples with `SequenceBatch`.
        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            indices = cast(SequenceData, example["indices"])

            seqs, seq_lens = indices["seqs"], indices["seq_lens"]
            target_mask = example["target_mask"]["seqs"]

            return SequenceBatch(
                seqs, seq_lens, target_mask=target_mask, example=example
            )

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline,
            gangs,
            num_accumulate=options.num_accumulate,
            drop_remainder=options.drop_remainder,
            sync=options.sync_batches,
            sync_mode=options.sync_mode,
        )


@dataclass
class LMSFTDataSource:
    path: str
    split: str | None = None
    weight: float = 1.0


@dataclass
class LMSFTDatasetConfig:
    sources: dict[str, list[LMSFTDataSource]] = field(default_factory=dict)


def open_lm_sft_dataset(config: LMSFTDatasetConfig) -> LMSFTDataset:
    return LMSFTDataset(config.sources)
