# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
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
from fairseq2.datasets import AbstractDatasetLoader, DataPipelineReader
from fairseq2.datasets.instruction import InstructionDataset, load_instruction_dataset
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import override

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
        shuffle_window_size: int = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        builder = list_files(self._data_dir, pattern="*.jsonl")

        # Shuffle the files. Must be consistent across all processes.
        builder.shuffle(shuffle_window=0, seed=seed)

        seed += 1

        builder.yield_from(partial(self._read_jsonl, tokenizer=tokenizer))

        builder.shuffle(shuffle_window_size, seed=seed)

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
        builder.shuffle(shuffle_window_size, seed=seed)

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
        def _example_to_batch(example: Dict[str, Any]) -> SequenceBatch:
            text = cast(SequenceData, example["tokens"])

            seqs, padding_mask = get_seqs_and_padding_mask(text, gang.device)

            target_mask = example["target_mask"]["seqs"].to(gang.device)

            return SequenceBatch(seqs, padding_mask, target_mask, example=example)

        pipeline = builder.map(_example_to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=True,
        )

    def _read_jsonl(self, path: str, tokenizer: TextTokenizer) -> DataPipeline:
        source_text_encoder = tokenizer.create_encoder(mode="prompt")
        target_text_encoder = tokenizer.create_encoder(mode="prompt_response")

        lines = []

        with Path(path).open() as fp:
            for line in fp:
                lines.append(line)

        builder = read_sequence(lines)

        builder.map(json.loads, num_parallel_calls=npc)

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
    def splits(self) -> Set[str]:
        return {"train"}


@final
class GenericInstructionDatasetLoader(AbstractDatasetLoader[GenericInstructionDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> GenericInstructionDataset:
        return GenericInstructionDataset(path)


load_generic_instruction_dataset = GenericInstructionDatasetLoader()


def _register_generic() -> None:
    load_instruction_dataset.register(
        "generic_instruction", load_generic_instruction_dataset
    )
