# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final, cast, final

from typing_extensions import override

from fairseq2.data import (
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    create_bucket_sizes,
)
from fairseq2.data.text import read_text
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DataReadError,
    DataReadOptions,
    DatasetHubAccessor,
    DatasetLoadError,
    LengthBatching,
    StaticBatching,
    UnknownSplitError,
)
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import Device


@dataclass(kw_only=True)
class ParallelTextReadOptions(DataReadOptions):
    direction: Direction | None = None
    """The direction to read. If ``None``, all directions will be read."""

    sample: bool = False
    """If ``True``, corpora will be sampled in proportion to their weights."""


@dataclass(unsafe_hash=True)  # Due to FSDP, we cannot freeze.
class Direction:
    """Represents the language direction of a parallel corpus."""

    source_lang: str
    """The source language code."""

    target_lang: str
    """The target language code."""

    origin: str | None = None
    """The origin of data. Typically used to indicate mined or synthetic data."""

    def __repr__(self) -> str:
        s = f"{self.source_lang}-{self.target_lang}"

        if self.origin:
            s = f"{self.origin}/{s}"

        return s


class ParallelTextDataset(ABC):
    """Represents a parallel text dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: ParallelTextReadOptions | None = None,
    ) -> DataReader[Seq2SeqBatch]:
        """Create a dataset reader.

        :param split:
            The split to read.
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

    @abstractmethod
    def splits(self) -> set[str]:
        """Return the set of splits."""

    @abstractmethod
    def directions(self, split: str) -> list[Direction]:
        """Return the directions included ``split``."""


# TODO: FIX, INFER
npc = 10


GENERIC_PARALLEL_TEXT_DATASET_FAMILY: Final = "generic_parallel_text"


@final
class GenericParallelTextDataset(ParallelTextDataset):
    """Represents a generic file-based parallel text dataset."""

    _name: str
    _data_dir: Path
    _splits: dict[str, tuple[list[Direction], list[float]]]

    def __init__(
        self,
        name: str,
        data_dir: Path,
        splits: dict[str, tuple[list[Direction], list[float]]],
    ) -> None:
        """
        :param data_dir:
            The directory under which the manifest and the language direction
            files reside.
        :param splits:
            The splits with their directions and their weights.
        """
        self._name = name

        for split, (directions, weights) in splits.items():
            if len(directions) != len(weights):
                raise ValueError(
                    f"The lengths of the direction and weight lists of the split '{split}' must match, but they are {len(directions)} and {len(weights)} instead."
                )

        self._data_dir = data_dir
        self._splits = splits

    @classmethod
    def from_path(cls, path: Path, name: str) -> GenericParallelTextDataset:
        path = path.expanduser().resolve()

        if not path.is_dir():
            raise DatasetLoadError(
                name, f"The '{path}' path of the '{name}' dataset is expected to be a directory with a MANIFEST file."  # fmt: skip
            )

        try:
            split_names = [d.name for d in path.iterdir() if d.is_dir()]
        except OSError as ex:
            raise DatasetLoadError(
                name, f"The splits under the '{path}' directory of the '{name}' dataset cannot be determined. See the nested exception for details."  # fmt: skip
            ) from ex

        splits = {}

        for split in split_names:
            manifest_file = path.joinpath(split).joinpath("MANIFEST")

            try:
                with manifest_file.open() as fp:
                    content = list(fp)
            except OSError as ex:
                raise DatasetLoadError(
                    name, f"The '{manifest_file}' file of the '{name}' dataset cannot be read. See the nested exception for details."  # fmt: skip
                ) from ex

            # Sort the directions in alphabetical order.
            content.sort()

            directions = []

            weights = []

            # Each line of the MANIFEST file corresponds to a direction and
            # its weight (e.g. number of examples) in the split.
            for idx, line in enumerate(content):

                def error() -> DatasetLoadError:
                    return DatasetLoadError(
                        name, f"Each line in the '{manifest_file}' manifest file of the '{name}' dataset must represent a valid direction and a weight, but line {idx} is '{line}' instead."  # fmt: skip
                    )

                fields = line.rstrip().split("\t")

                if len(fields) != 2:
                    raise error()

                try:
                    direction = cls._parse_direction(fields[0])
                except ValueError:
                    raise error() from None

                directions.append(direction)

                try:
                    weight = float(fields[1].strip())
                except ValueError:
                    raise error() from None

                weights.append(weight)

            splits[split] = (directions, weights)

        return GenericParallelTextDataset(name, data_dir=path, splits=splits)

    @staticmethod
    def _parse_direction(s: str) -> Direction:
        def value_error() -> ValueError:
            return ValueError(
                f"`s` must represent a valid direction, but is '{s}' instead."
            )

        parts = s.rstrip().split("/")

        if len(parts) == 1:
            origin, lang_pair = None, parts[0]
        elif len(parts) == 2:
            origin, lang_pair = parts
        else:
            raise value_error()

        parts = lang_pair.split("-")

        if len(parts) != 2:
            raise value_error()

        source_lang, target_lang = parts

        source_lang = source_lang.strip()
        target_lang = target_lang.strip()

        if not source_lang or not target_lang:
            raise value_error()

        return Direction(source_lang, target_lang, origin)

    @override
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: ParallelTextReadOptions | None = None,
    ) -> DataPipelineReader[Seq2SeqBatch]:
        directions_weights = self._splits.get(split)
        if directions_weights is None:
            raise UnknownSplitError(self._name, split, self._splits.keys())

        if options is None:
            options = ParallelTextReadOptions()

        seed = options.seed

        directions, weights = directions_weights

        # Determine the directions to read.
        direction = options.direction

        if direction is not None:
            if direction not in directions:
                raise ValueError(
                    f"`direction` must be a direction that exists in '{split}' split, but is '{direction}' instead."
                )

            directions = [direction]

        # Initialize the text encoders for each direction.
        text_encoders = {}

        for direction in directions:
            source_mode = "source"

            if direction.origin:
                source_mode = f"{source_mode}_{direction.origin}"

            source_encoder = tokenizer.create_encoder(
                task="translation", lang=direction.source_lang, mode=source_mode
            )

            target_encoder = tokenizer.create_encoder(
                task="translation", lang=direction.target_lang, mode="target"
            )

            text_encoders[direction] = (source_encoder, target_encoder)

        if len(directions) == 1:
            builder = self._read_direction(split, directions[0])
        else:
            # Build the direction pipelines.
            pipelines = []

            for direction in directions:
                pipeline = self._read_direction(split, direction).and_return()

                pipelines.append(pipeline)

            if options.sample:
                builder = DataPipeline.sample(pipelines, weights=weights, seed=seed)

                seed += 1
            else:
                builder = DataPipeline.concat(pipelines)

        # Shuffle examples. Must be consistent across all processes.
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        # Encode source and target texts.
        def encode(example: dict[str, Any]) -> dict[str, Any]:
            direction = example["direction"]

            source_encoder, target_encoder = text_encoders[direction]

            example["source_indices"] = source_encoder(example["source_text"])
            example["target_indices"] = target_encoder(example["target_text"])

            return example

        builder.map(encode, num_parallel_calls=npc)

        batching = options.batching

        if isinstance(batching, LengthBatching):
            bucket_sizes = create_bucket_sizes(
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                max_num_elements=batching.max_num_elements,
            )

            # Bucket by the length of the source or target sequence. The longer
            # one will be considered the length of the example.
            builder.bucket_by_length(
                bucket_sizes,
                selector="source_indices,target_indices",
                min_data_len=min_seq_len,
                skip_below_min_examples=True,
                skip_above_max_examples=True,
                drop_remainder=options.drop_remainder,
            )
        elif isinstance(batching, StaticBatching):
            # Filter out out-of-range examples.
            def skip(example: dict[str, Any]) -> bool:
                source_len = len(example["source_indices"])
                target_len = len(example["target_indices"])

                max_len = max(source_len, target_len)

                return max_len >= min_seq_len and max_len <= max_seq_len

            builder.filter(skip)

            # Bucket `batch_size` examples.
            builder.bucket(batching.batch_size, drop_remainder=options.drop_remainder)
        else:
            raise NotSupportedError(f"`{batching}` is not supported.")

        # Shuffle buckets.
        if options.batch_shuffle_window != 1:
            builder.shuffle(options.batch_shuffle_window, seed)

        seed += 1

        # Collate bucketed examples into a batch.
        collater = Collater(pad_value=tokenizer.vocab_info.pad_idx)

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(options.num_prefetch)

        f = partial(self._to_batch, device=gang.device)

        pipeline = builder.map(f).and_return()

        return DataPipelineReader[Seq2SeqBatch](
            self._name, split, pipeline, gang, options
        )

    def _read_direction(self, split: str, direction: Direction) -> DataPipelineBuilder:
        direction_pipeline = DataPipeline.constant(direction).and_return()

        source_file = self._data_dir.joinpath(split)
        target_file = self._data_dir.joinpath(split)

        if direction.origin is not None:
            source_file = source_file.joinpath(direction.origin)
            target_file = target_file.joinpath(direction.origin)

        source = direction.source_lang
        target = direction.target_lang

        source_file = source_file.joinpath(f"{source}-{target}.{source}.txt")
        target_file = target_file.joinpath(f"{source}-{target}.{target}.txt")

        if not source_file.exists():
            raise DataReadError(
                self._name, split, f"The '{source_file}' file is not found under the '{self._data_dir}' directory of the '{self._name}' dataset."  # fmt: skip
            )

        if not target_file.exists():
            raise DataReadError(
                self._name, split, f"The '{target_file}' file is not found under the '{self._data_dir}' directory of the '{self._name}' dataset."  # fmt: skip
            )

        source_builder = read_text(source_file, rtrim=True, memory_map=True)
        target_builder = read_text(target_file, rtrim=True, memory_map=True)

        source_pipeline = source_builder.and_return()
        target_pipeline = target_builder.and_return()

        pipelines = [direction_pipeline, source_pipeline, target_pipeline]

        return DataPipeline.zip(
            pipelines, names=["direction", "source_text", "target_text"]
        )

    @staticmethod
    def _to_batch(example: dict[str, Any], device: Device) -> Seq2SeqBatch:
        source_data = cast(SequenceData, example["source_indices"])
        target_data = cast(SequenceData, example["target_indices"])

        source_seqs, source_padding_mask = get_seqs_and_padding_mask(
            source_data, device
        )
        target_seqs, target_padding_mask = get_seqs_and_padding_mask(
            target_data, device
        )

        return Seq2SeqBatch(
            source_seqs,
            source_padding_mask,
            target_seqs,
            target_padding_mask,
            example,
        )

    @override
    def splits(self) -> set[str]:
        return set(self._splits.keys())

    @override
    def directions(self, split: str) -> list[Direction]:
        directions_weights = self._splits.get(split)
        if directions_weights is None:
            raise UnknownSplitError(self._name, split, self._splits.keys())

        return directions_weights[0]


get_parallel_text_dataset_hub = DatasetHubAccessor(ParallelTextDataset)
