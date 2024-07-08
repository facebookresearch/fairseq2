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
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast, final

from typing_extensions import NoReturn

from fairseq2.assets import AssetCard, AssetError
from fairseq2.data import (
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    create_bucket_sizes,
)
from fairseq2.data.text import TextTokenizer, read_text
from fairseq2.datasets.batching import LengthBatching, StaticBatching
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.error import DatasetError
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import Device, override


@dataclass(frozen=True)
class Direction:
    """Represents the language direction of a parallel corpus."""

    source_lang: str
    """The source language code."""

    target_lang: str
    """The target language code."""

    origin: Optional[str] = None
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
        max_seq_len: int,
        batching: Union[StaticBatching, LengthBatching],
        *,
        direction: Optional[Direction] = None,
        min_seq_len: int = 1,
        sample: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataReader[Seq2SeqBatch]:
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
        :param batching:
            The batching strategy for returned examples.
        :param direction:
            The direction to read. If ``None``, all directions will be read.
        :param min_seq_len:
            The minimum sequence length of each example. Examples shorter than
            this value will be dropped.
        :param sample:
            If ``True``, corpora will be sampled in proportion to their weights.
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
    def splits(self) -> Set[str]:
        """Return the set of splits."""

    @abstractmethod
    def directions(self, split: str) -> List[Direction]:
        """Return the directions included ``split``."""


load_parallel_text_dataset = DelegatingDatasetLoader[ParallelTextDataset]()


# TODO: FIX, INFER
npc = 10


@final
class GenericParallelTextDataset(ParallelTextDataset):
    """Represents a generic file-based parallel text dataset."""

    _data_dir: Path
    _splits: Dict[str, Tuple[List[Direction], List[float]]]

    def __init__(
        self,
        *,
        data_dir: Path,
        splits: Dict[str, Tuple[List[Direction], List[float]]],
    ) -> None:
        """
        :param data_dir:
            The directory under which the manifest and the language direction
            files reside.
        :param splits:
            The splits with their directions and their weights.
        """
        for split, (directions, weights) in splits.items():
            if len(directions) != len(weights):
                raise ValueError(
                    f"The lengths of the direction and weight lists of the split '{split}' must match, but they are {len(directions)} and {len(weights)} instead."
                )

        self._data_dir = data_dir
        self._splits = splits

    @classmethod
    def from_path(cls, path: Path) -> GenericParallelTextDataset:
        """Load a :class:`GenericParallelTextDataset` from ``path``."""
        path = path.expanduser().resolve()

        if not path.is_dir():
            raise ValueError("`path` must be a directory with a MANIFEST file.")

        try:
            split_names = [d.name for d in path.iterdir() if d.is_dir()]
        except OSError as ex:
            raise RuntimeError(
                "The splits cannot be determined. See nested exception for details."
            ) from ex

        splits = {}

        for split in split_names:
            manifest_file = path.joinpath(split).joinpath("MANIFEST")

            try:
                fp = manifest_file.open()
            except OSError as ex:
                raise RuntimeError(
                    f"{manifest_file} cannot be read. See nested exception for details."
                ) from ex

            try:
                content = list(fp)
            except OSError as ex:
                raise RuntimeError(
                    f"{manifest_file} cannot be read. See nested exception for details."
                ) from ex
            finally:
                fp.close()

            # Sort the directions in alphabetical order.
            content.sort()

            directions = []

            weights = []

            # Each line of the MANIFEST file corresponds to a direction and
            # its weight (e.g. number of examples) in the split.
            for idx, line in enumerate(content):

                def raise_error() -> NoReturn:
                    raise DatasetError(
                        f"Each line in {manifest_file} must represent a valid direction and a weight, but line {idx} is '{line}' instead."
                    )

                fields = line.rstrip().split("\t")

                if len(fields) != 2:
                    raise_error()

                try:
                    direction = cls._parse_direction(fields[0])
                except ValueError:
                    raise_error()

                directions.append(direction)

                try:
                    weight = float(fields[1].strip())
                except ValueError:
                    raise_error()

                weights.append(weight)

            splits[split] = (directions, weights)

        return GenericParallelTextDataset(data_dir=path, splits=splits)

    @staticmethod
    def _parse_direction(s: str) -> Direction:
        def raise_error() -> NoReturn:
            raise ValueError(
                f"`s` must represent a valid direction, but is '{s}' instead."
            )

        parts = s.rstrip().split("/")

        if len(parts) == 1:
            origin, lang_pair = None, parts[0]
        elif len(parts) == 2:
            origin, lang_pair = parts
        else:
            raise_error()

        parts = lang_pair.split("-")

        if len(parts) != 2:
            raise_error()

        source_lang, target_lang = parts

        source_lang = source_lang.strip()
        target_lang = target_lang.strip()

        if not source_lang or not target_lang:
            raise_error()

        return Direction(source_lang, target_lang, origin)

    @override
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        batching: Union[StaticBatching, LengthBatching],
        *,
        direction: Optional[Direction] = None,
        min_seq_len: int = 1,
        sample: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataPipelineReader[Seq2SeqBatch]:
        try:
            directions, weights = self._splits[split]
        except KeyError:
            self._raise_split_error(split)

        # Determine the directions to read.
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

            if sample:
                builder = DataPipeline.sample(pipelines, weights=weights, seed=seed)

                seed += 1
            else:
                builder = DataPipeline.concat(pipelines)

        # Shuffle examples. Must be consistent across all processes.
        if example_shuffle_window != 1:
            builder.shuffle(example_shuffle_window, seed)

        seed += 1

        static_batching = isinstance(batching, StaticBatching)

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=not static_batching)

        seed += gang.rank

        # Encode source and target texts.
        def encode(example: Dict[str, Any]) -> Dict[str, Any]:
            direction = example["direction"]

            source_encoder, target_encoder = text_encoders[direction]

            example["source_indices"] = source_encoder(example["source_text"])
            example["target_indices"] = target_encoder(example["target_text"])

            return example

        builder.map(encode, num_parallel_calls=npc)

        if isinstance(batching, LengthBatching):
            # Bucket by the length of the source or target sequence. The longer
            # one will be considered the length of the example.
            bucket_sizes = create_bucket_sizes(
                max_seq_len=max_seq_len,
                min_seq_len=min_seq_len,
                max_num_elements=batching.max_num_elements,
            )

            builder.bucket_by_length(
                bucket_sizes,
                selector="source_indices,target_indices",
                min_data_len=min_seq_len,
                skip_below_min_examples=True,
                skip_above_max_examples=True,
            )
        else:
            # Filter out out-of-range examples.
            def skip(example: Dict[str, Any]) -> bool:
                source_len = len(example["source_indices"])
                target_len = len(example["target_indices"])

                max_len = max(source_len, target_len)

                return max_len >= min_seq_len and max_len <= max_seq_len

            builder.filter(skip)

            # Bucket `batch_size` examples.
            builder.bucket(batching.batch_size)

        # Shuffle buckets.
        if batch_shuffle_window != 1:
            builder.shuffle(batch_shuffle_window, seed)

        seed += 1

        # Collate bucketed examples into a batch.
        collater = Collater(pad_value=tokenizer.vocab_info.pad_idx)

        builder.map(collater, num_parallel_calls=npc)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(num_prefetch)

        f = partial(self._to_batch, device=gang.device)

        pipeline = builder.map(f).and_return()

        return DataPipelineReader[Seq2SeqBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=not static_batching,
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
            raise DatasetError(
                f"The source file '{source_file}' is not found under {self._data_dir}."
            )

        if not target_file.exists():
            raise DatasetError(
                f"The target file '{target_file}' is not found under {self._data_dir}."
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
    def _to_batch(example: Dict[str, Any], device: Device) -> Seq2SeqBatch:
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
    def splits(self) -> Set[str]:
        return set(self._splits.keys())

    @override
    def directions(self, split: str) -> List[Direction]:
        try:
            directions, _ = self._splits[split]
        except KeyError:
            self._raise_split_error(split)

        return directions

    def _raise_split_error(self, split: str) -> NoReturn:
        raise ValueError(
            f"`split` must be one of the following splits, but is '{split}' instead: {', '.join(sorted(self._splits.keys()))}"
        )


@final
class GenericParallelTextDatasetLoader(
    AbstractDatasetLoader[GenericParallelTextDataset]
):
    @override
    def _load(self, path: Path, card: AssetCard) -> GenericParallelTextDataset:
        try:
            return GenericParallelTextDataset.from_path(path)
        except RuntimeError as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex


load_generic_parallel_text_dataset = GenericParallelTextDatasetLoader()

load_parallel_text_dataset.register(
    "generic_parallel_text", load_generic_parallel_text_dataset
)
