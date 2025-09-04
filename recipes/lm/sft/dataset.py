# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, final

from torch import Tensor

from fairseq2.data.data_pipeline import (
    DataPipeline,
    read_sequence,
)
from fairseq2.data.text import read_text
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DatasetOpenError,
    SequenceBatch,
    SyncMode,
)
from fairseq2.gang import Gangs
from fairseq2.nn import BatchLayout

# TODO: FIX, INFER
npc = 10


LM_SFT_DATASET: Final = "lm_sft"

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

@final
class LMSFTDataset:
    _name: str
    _files: Sequence[Path]

    def __init__(self, name: str, files: Sequence[Path]) -> None:
        self._name = name
        self._files = files

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

        # Prefetch `prefetch` batches in background.
        builder.prefetch(options.prefetch)

        # Wrap examples with `SequenceBatch`.
        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            indices = cast(SequenceData, example["indices"])

            seqs, seq_lens = indices["seqs"], indices["seq_lens"]

            return SequenceBatch(seqs, seq_lens, example=example)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            self._name, split, pipeline, gang, options
        )


@dataclass
class LMSFTDatasetConfig:
    path: Path = field(default_factory=Path)


def open_lm_sft_dataset(name: str, config: LMSFTDatasetConfig) -> LMSFTDataset:
    path = config.path

    path = path.expanduser().resolve()

    if not path.is_dir():
        files = [path]
    else:
        try:
            files = [f for f in path.glob("**/*.chunk.*.jsonl") if not f.is_dir()]
        except OSError as ex:
            raise DatasetOpenError(
                name, f"The text files under the '{path}' directory of the '{name}' dataset cannot be retrieved. See the nested exception for details."  # fmt: skip
            ) from ex

        files.sort()

    return LMSFTDataset(name, files)
