# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, cast, final

import torch
from typing_extensions import NoReturn

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
from fairseq2.datasets.batching import LengthBatching, StaticBatching
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.error import DatasetError
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
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        batching: Union[StaticBatching, LengthBatching],
        *,
        sample: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        max_num_batches: Optional[int] = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataReader[SequenceBatch]:
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
        :param max_num_batches:
            The maximum number of batches to return.
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
    def create_prompt_reader(
        self,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        batching: StaticBatching,
        *,
        num_prefetch: int = 1,
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        """Create a dataset reader for evaluation.

        :param tokenizer:
            The tokenizer to encode text.
        :param gang:
            The gang over which to shard the dataset.
        :param max_seq_len:
            The maximum sequence length of each example. Examples longer than
            this value will be dropped.
        :param batching:
            The batching strategy for returned examples.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param extras:
            The extra parameters specific to the dataset implementation.
        """


load_instruction_dataset = DelegatingDatasetLoader[InstructionDataset]()


# TODO: FIX, INFER
npc = 10


# TODO: Work in progress!
@final
class GenericInstructionDataset(InstructionDataset):
    """Represents a generic JSONL instruction dataset."""

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
    def from_path(cls, path: Path) -> GenericInstructionDataset:
        """Load a :class:`InstructionDataset` from ``path``."""
        path = path.expanduser().resolve()

        if not path.is_dir():
            return GenericInstructionDataset(files=[path], weights=[1.0])

        manifest_file = path.joinpath("MANIFEST")

        try:
            fp = manifest_file.open()
        except FileNotFoundError:
            fp = None
        except OSError as ex:
            raise RuntimeError(
                f"{manifest_file} cannot be read. See nested exception for details."
            ) from ex

        # If the directory does not contain a MANIFEST file, treat all JSONL
        # files as part of the dataset with equal weight.
        if fp is None:
            try:
                files = list(path.glob("**/*.jsonl"))
            except OSError as ex:
                raise RuntimeError(
                    f"The JSONL files under {path} cannot be retrieved. See nested exception for details."
                ) from ex

            weights = [1.0 for _ in range(len(files))]

            return GenericInstructionDataset(files, weights=weights)

        try:
            content = list(fp)
        except OSError as ex:
            raise RuntimeError(
                f"{manifest_file} cannot be read. See nested exception for details."
            ) from ex
        finally:
            fp.close()

        # Sort the JSONL files in alphabetical order.
        content.sort()

        files = []

        weights = []

        # Each line of the MANIFEST file corresponds to the path of a JSONL file
        # and its weight (e.g. number of examples).
        for idx, line in enumerate(content):

            def raise_error() -> NoReturn:
                raise DatasetError(
                    f"Each line in {manifest_file} must represent a path to a JSONL file and a weight, but line {idx} is '{line}' instead."
                )

            fields = line.rstrip().split("\t")

            if len(fields) != 2:
                raise_error()

            file_path = fields[0].strip()
            if not file_path:
                raise_error()

            try:
                file = path.joinpath(file_path)
            except ValueError:
                raise_error()

            if not file.exists():
                raise DatasetError(
                    f"The file '{file}' referred at line {idx} in {manifest_file} does not exist."
                )

            files.append(file)

            try:
                weight = float(fields[1].strip())
            except ValueError:
                raise_error()

            weights.append(weight)

        return GenericInstructionDataset(files, weights)

    @override
    def create_reader(
        self,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        batching: Union[StaticBatching, LengthBatching],
        *,
        sample: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        max_num_batches: Optional[int] = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
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

        static_batching = isinstance(batching, StaticBatching)

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=not static_batching)

        seed += gang.rank

        # Encode prompt and target texts.
        prompt_encoder = tokenizer.create_encoder(mode="prompt")
        target_encoder = tokenizer.create_encoder(mode="prompt_response")

        builder.map(prompt_encoder, selector="src", num_parallel_calls=npc)
        builder.map(target_encoder, selector="tgt", num_parallel_calls=npc)

        def cat_source_and_target(example: Dict[str, Any]) -> Dict[str, Any]:
            prompt_indices = example["src"]
            target_indices = example["tgt"]

            indices = torch.cat([prompt_indices, target_indices])

            target_mask = torch.arange(len(indices)) >= len(prompt_indices)

            return {"indices": indices, "target_mask": target_mask}

        builder.map(cat_source_and_target, num_parallel_calls=npc)

        if isinstance(batching, LengthBatching):
            bucket_sizes = create_bucket_sizes(
                max_seq_len=max_seq_len, max_num_elements=batching.max_num_elements
            )

            # Bucket by the sequence length.
            builder.bucket_by_length(
                bucket_sizes, selector="indices", skip_above_max_examples=True
            )
        else:
            # Filter out long examples.
            def skip(example: Dict[str, Any]) -> bool:
                return len(example["indices"]) <= max_seq_len

            builder.filter(skip)

            # Bucket `batch_size` examples.
            builder.bucket(batching.batch_size)

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
        def to_batch(example: Dict[str, Any]) -> SequenceBatch:
            indices = cast(SequenceData, example["indices"])

            seqs, padding_mask = get_seqs_and_padding_mask(indices, gang.device)

            target_mask = example["target_mask"]["seqs"].to(gang.device)

            return SequenceBatch(seqs, padding_mask, target_mask, example=example)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=not static_batching,
        )

    @override
    def create_prompt_reader(
        self,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        batching: StaticBatching,
        *,
        num_prefetch: int = 1,
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        if len(self._files) == 1:
            builder = self._read_jsonl(self._files[0], tokenizer)
        else:
            pipelines = []

            for file in self._files:
                pipeline = self._read_jsonl(file, tokenizer).and_return()

                pipelines.append(pipeline)

            builder = DataPipeline.concat(pipelines)

        # Shard
        builder.shard(gang.rank, gang.size)

        # Encode prompt texts.
        text_encoder = tokenizer.create_encoder(mode="prompt")

        def encode(example: Dict[str, Any]) -> Dict[str, Any]:
            prompt = example["src"]

            return {"prompt": prompt, "indices": text_encoder(prompt)}

        builder.map(encode, num_parallel_calls=npc)

        # Filter out long examples.
        def skip(example: Dict[str, Any]) -> bool:
            return len(example["indices"]) <= max_seq_len

        builder.filter(skip)

        # Bucket `batch_size` examples.
        builder.bucket(batching.batch_size)

        # Collate bucketed examples into a batch.
        collater = Collater(pad_value=tokenizer.vocab_info.pad_idx or 0)

        builder.map(collater, num_parallel_calls=npc)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(num_prefetch)

        # Wrap examples with `SequenceBatch`.
        def to_batch(example: Dict[str, Any]) -> SequenceBatch:
            indices = cast(SequenceData, example["indices"])

            seqs, padding_mask = get_seqs_and_padding_mask(indices, gang.device)

            return SequenceBatch(seqs, padding_mask, example=example)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](pipeline, gang)

    def _read_jsonl(self, path: Path, tokenizer: TextTokenizer) -> DataPipelineBuilder:
        lines = []

        # TODO(balioglu): Do in C++.
        with path.open() as fp:
            for line in fp:
                lines.append(line)

        return read_sequence(lines).map(json.loads, num_parallel_calls=npc)


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
