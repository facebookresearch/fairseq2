from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
from torch import Tensor

@dataclass
class PreferenceOptimizationBatch:
    """Represents a preference optimization batch."""
    chosen: SequenceBatch
    rejected: SequenceBatch


class PreferenceOptimizationDataset(ABC):
    """Represents an preference optimization finetuning dataset."""

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

load_preference_optimization_dataset = DelegatingDatasetLoader[PreferenceOptimizationDataset]()

# TODO: FIX, INFER
npc = 10

@final
class GenericPreferenceOptimizationDataset(PreferenceOptimizationDataset):
    """Represents a generic JSONL preferemce preference optimization dataset."""

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
    def from_path(cls, path: Path) -> PreferenceOptimizationDataset:
        """Load a :class:`PreferenceOptimizationDataset` from ``path``."""
        path = path.expanduser().resolve()

        if not path.is_dir():
            return GenericPreferenceOptimizationDataset(files=[path], weights=[1.0])

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

            return GenericPreferenceOptimizationDataset(files, weights=weights)

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

        return GenericPreferenceOptimizationDataset(files, weights)

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

        static_batching = isinstance(batching, StaticBatching)

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=not static_batching)

        seed += gang.rank

        if isinstance(batching, LengthBatching):
            bucket_sizes = create_bucket_sizes(
                max_seq_len=max_seq_len, max_num_elements=batching.max_num_elements
            )

            # Bucket by the sequence length.
            builder.bucket_by_length(
                bucket_sizes, selector="chosen_tokens", skip_above_max_examples=True
            )
        else:
            # Filter out long examples.
            def skip(example: Dict[str, Any]) -> bool:
                return len(example["chosen_tokens"]) <= max_seq_len

            builder.filter(skip)

            # Bucket `batch_size` examples.
            builder.bucket(batching.batch_size)

        # Shuffle buckets.
        if batch_shuffle_window != 1:
            builder.shuffle(batch_shuffle_window, seed=seed)

        seed += 1

        # Collate bucketed examples into a batch.
        target_mask_collate_opts = [
            CollateOptionsOverride("chosen_target_mask", pad_value=False),
            CollateOptionsOverride("rejected_target_mask", pad_value=False),
        ]

        collater = Collater(pad_value=0, overrides=target_mask_collate_opts)

        builder.map(collater, num_parallel_calls=npc)

        # Prefetch `num_prefetch` examples in background.
        builder.prefetch(num_prefetch)

        # Wrap examples with `DpoInstructionBatch`.
        def _example_to_batch(example: Dict[str, Any]) -> PreferenceOptimizationBatch:
            chosen_text = cast(SequenceData, example["chosen_tokens"])
            rejected_text = cast(SequenceData, example["rejected_tokens"])

            chosen_seqs, chosen_padding_mask = get_seqs_and_padding_mask(
                chosen_text, gang.device
            )
            rejected_seqs, rejected_padding_mask = get_seqs_and_padding_mask(
                rejected_text, gang.device
            )

            chosen_target_mask = example["chosen_target_mask"]["seqs"].to(gang.device)
            rejected_target_mask = example["rejected_target_mask"]["seqs"].to(
                gang.device
            )

            chosen_batch = SequenceBatch(
                chosen_seqs,
                chosen_padding_mask,
                chosen_target_mask,
                example=example["chosen_tokens"],
            )

            rejected_batch = SequenceBatch(
                rejected_seqs,
                rejected_padding_mask,
                rejected_target_mask,
                example=example["rejected_tokens"],
            )

            return PreferenceOptimizationBatch(chosen_batch, rejected_batch)

        pipeline = builder.map(_example_to_batch).and_return()

        return DataPipelineReader[PreferenceOptimizationBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=not static_batching,
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
        builder.map(target_text_encoder, selector="tgt_chosen", num_parallel_calls=npc)
        builder.map(
            target_text_encoder, selector="tgt_rejected", num_parallel_calls=npc
        )

        def cat_source_and_text(d: Dict[str, Any]) -> Dict[str, Tensor]:
            source_tokens = d["src"]
            chosen_target_tokens = d["tgt_chosen"]
            rejected_target_tokens = d["tgt_rejected"]

            chosen_tokens = torch.cat([source_tokens, chosen_target_tokens])
            rejected_tokens = torch.cat([source_tokens, rejected_target_tokens])

            chosen_target_mask = torch.arange(len(chosen_tokens)) >= len(source_tokens)
            rejected_target_mask = torch.arange(len(rejected_tokens)) >= len(
                source_tokens
            )

            return {
                "chosen_tokens": chosen_tokens,
                "chosen_target_mask": chosen_target_mask,
                "rejected_tokens": rejected_tokens,
                "rejected_target_mask": rejected_target_mask,
            }

        builder.map(cat_source_and_text, num_parallel_calls=npc)

        return builder.and_return()


@final
class GenericPreferenceOptimizationDatasetLoader(AbstractDatasetLoader[GenericPreferenceOptimizationDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> GenericPreferenceOptimizationDataset:
        try:
            return GenericPreferenceOptimizationDataset.from_path(path)
        except RuntimeError as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex


load_generic_preference_optimization_dataset = GenericPreferenceOptimizationDatasetLoader()

load_preference_optimization_dataset.register(
    "generic_preference_optimization", load_generic_preference_optimization_dataset
)
