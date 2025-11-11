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
from typing_extensions import override

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
from fairseq2.datasets import DataPipelineReader, SequenceBatch
from fairseq2.device import Device, SupportsDeviceTransfer
from fairseq2.error import NotSupportedError, raise_operational_system_error
from fairseq2.gang import Gangs
from fairseq2.utils.uri import Uri

from ..common import DataReadOptions, LengthBatching, StaticBatching


LM_DPO_DATASET: Final = "lm_dpo"


@dataclass
class PreferenceBatch(SupportsDeviceTransfer):
    """Represents a preference optimization dataset batch."""

    chosen: SequenceBatch
    rejected: SequenceBatch
    reference_score_chosen: torch.Tensor | None
    reference_score_rejected: torch.Tensor | None

    @property
    def batch_size(self) -> int:
        """The size of the batch dimension."""
        return self.chosen.batch_size

    @override
    def to(self, device: Device, *, non_blocking: bool = False) -> None:
        self.chosen.to(device, non_blocking=non_blocking)

        self.rejected.to(device, non_blocking=non_blocking)

        if self.reference_score_chosen is not None:
            self.reference_score_chosen = self.reference_score_chosen.to(
                device, non_blocking=non_blocking
            )

        if self.reference_score_rejected is not None:
            self.reference_score_rejected = self.reference_score_rejected.to(
                device, non_blocking=non_blocking
            )


@dataclass(kw_only=True)
class LMDPODataReadOptions(DataReadOptions):
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

    chat_mode: bool = False


class LMDPODataset:
    def __init__(self, sources: dict[str, list[LMDPODataSource]]) -> None:
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
        options: LMDPODataReadOptions | None = None,
    ) -> DataPipelineReader[PreferenceBatch]:
        if options is None:
            options = LMDPODataReadOptions()

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
                    id_ = example.get("id")
                    chat_chosen = example.get("chat_chosen")
                    chat_rejected = example.get("chat_rejected")

                    encoded_output_chosen = encoder.apply_chat_template(
                        chat_chosen,
                        return_dict=True,
                        return_assistant_tokens_mask=True,
                        return_tensors="pt",
                    )

                    indices_chosen = encoded_output_chosen["input_ids"][0]
                    target_mask_chosen = encoded_output_chosen["assistant_masks"][
                        0
                    ].bool()

                    encoded_output_rejected = encoder.apply_chat_template(
                        chat_rejected,
                        return_dict=True,
                        return_assistant_tokens_mask=True,
                        return_tensors="pt",
                    )

                    indices_rejected = encoded_output_rejected["input_ids"][0]
                    target_mask_rejected = encoded_output_rejected["assistant_masks"][0].bool()
                    
                    breakpoint()
                    if not options.mask_source_tokens:
                        # no source masking i.e. mask has all 1s
                        target_mask_chosen._fill(True)
                        target_mask_rejected._fill(True)

                    total_tokens = len(indices_chosen) + len(indices_rejected)
                    
                    return {
                        "id": id_,
                        "indices_chosen": indices_chosen,
                        "indices_rejected": indices_rejected,
                        "reference_score_chosen": example.get(
                            "reference_score_chosen", None
                        ),
                        "reference_score_rejected": example.get(
                            "reference_score_rejected", None
                        ),
                        "target_mask_chosen": target_mask_chosen,
                        "target_mask_rejected": target_mask_rejected,
                        "total_tokens": total_tokens,
                    }

                builder.map(encoding_chat)

        else:
            # Encode source and target texts.
            source_encoder = tokenizer.create_encoder(mode=options.source_encode_mode)
            target_encoder = tokenizer.create_encoder(mode=options.target_encode_mode)

            builder.map(source_encoder, selector="src")
            builder.map(target_encoder, selector="tgt_chosen")
            builder.map(target_encoder, selector="tgt_rejected")
            
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
                    target_mask_rejected = (
                        torch.arange(len(indices_rejected)) >= source_len
                    )
                else:
                    target_mask_chosen = torch.full([len(indices_chosen)], True)
                    target_mask_rejected = torch.full([len(indices_rejected)], True)

                total_tokens = (
                    2 * len(source_indices)
                    + len(target_indices_chosen)
                    + len(target_indices_rejected)
                )

                return {
                    "id": id_,
                    "indices_prompt": source_indices,
                    "indices_chosen": indices_chosen,
                    "indices_rejected": indices_rejected,
                    "reference_score_chosen": example.get(
                        "reference_score_chosen", None
                    ),
                    "reference_score_rejected": example.get(
                        "reference_score_rejected", None
                    ),
                    "target_mask_chosen": target_mask_chosen,
                    "target_mask_rejected": target_mask_rejected,
                    "total_tokens": total_tokens,
                }

            builder.map(cat_source_and_target)

        batching = options.batching

        if isinstance(batching, LengthBatching):
            bucket_sizes = create_bucket_sizes(
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                max_num_elements=batching.max_num_elements,
            )

            # Bucket by the sequence length
            builder.bucket_by_length(
                bucket_sizes,
                selector="total_tokens",
                min_data_len=min_seq_len,
                skip_above_max_examples=True,
                drop_remainder=options.drop_remainder,
            )
        elif isinstance(batching, StaticBatching):
            # Filter out long examples
            def skip(example: dict[str, Any]) -> bool:
                chosen_len = len(example["indices_chosen"])
                rejected_len = len(example["indices_rejected"])

                if chosen_len > max_seq_len or rejected_len > max_seq_len:
                    return False

                return chosen_len >= min_seq_len and rejected_len >= min_seq_len

            builder.filter(skip)

            # Bucket `batch_size` examples
            builder.bucket(batching.batch_size, drop_remainder=options.drop_remainder)
        else:
            raise NotSupportedError(f"`{batching}` is not supported.")

        # Shuffle buckets
        if options.batch_shuffle_window != 1:
            builder.shuffle(options.batch_shuffle_window, seed=seed)

        seed += 1
        
        # Collate bucketed examples into a batch.
        target_mask_collate_opts = [
            CollateOptionsOverride("target_mask_chosen", pad_value=False),
            CollateOptionsOverride("target_mask_rejected", pad_value=False),
        ]

        collater = Collater(pad_value=0, overrides=target_mask_collate_opts)

        builder.map(collater, num_parallel_calls=options.npc)
        
        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Prefetch `prefetch` batches in background.
        builder.prefetch(options.prefetch)
        
        # Wrap examples with `PreferenceBatch`.
        def to_batch(example: dict[str, Any]) -> PreferenceBatch:
            indices_chosen = cast(SequenceData, example["indices_chosen"])
            indices_rejected = cast(SequenceData, example["indices_rejected"])

            seqs_chosen, seq_chosen_lens = (
                indices_chosen["seqs"],
                indices_chosen["seq_lens"],
            )
            seqs_rejected, seq_rejected_lens = (
                indices_rejected["seqs"],
                indices_rejected["seq_lens"],
            )

            target_mask_chosen = example["target_mask_chosen"]["seqs"]
            target_mask_rejected = example["target_mask_rejected"]["seqs"]

            batch_chosen = SequenceBatch(
                seqs_chosen,
                seq_chosen_lens,
                target_mask=target_mask_chosen,
                example=example,
            )

            batch_rejected = SequenceBatch(
                seqs_rejected,
                seq_rejected_lens,
                target_mask=target_mask_rejected,
                example=example,
            )

            batch_reference_scores_chosen = None
            if all(example["reference_score_chosen"]):
                batch_reference_scores_chosen = torch.Tensor(
                    example["reference_score_chosen"]
                )
            batch_reference_scores_rejected = None
            if all(example["reference_score_rejected"]):
                batch_reference_scores_rejected = torch.Tensor(
                    example["reference_score_rejected"]
                )

            return PreferenceBatch(
                batch_chosen,
                batch_rejected,
                batch_reference_scores_chosen,
                batch_reference_scores_rejected,
            )

        pipeline = builder.map(to_batch).and_return()
        
        return DataPipelineReader[PreferenceBatch](
            pipeline,
            gangs,
            num_accumulate=options.num_accumulate,
            drop_remainder=options.drop_remainder,
            sync=options.sync_batches,
            sync_mode=options.sync_mode,
        )


@dataclass
class LMDPODataSource:  #TODO: to confirm after recipe.py
    path: str
    split: str | None = None
    weight: float = 1.0


@dataclass
class LMDPODatasetConfig:  #TODO: to confirm after recipe.py
    sources: dict[str, list[LMDPODataSource]] = field(default_factory=dict)


def open_lm_dpo_dataset(config: LMDPODatasetConfig) -> LMDPODataset:
    return LMDPODataset(config.sources)
