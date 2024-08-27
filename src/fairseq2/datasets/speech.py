# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, final

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import layer_norm
from typing_extensions import override

from fairseq2.assets import AssetCard, AssetError
from fairseq2.data import Collater, DataPipelineBuilder, FileMapper, read_sequence
from fairseq2.data.audio import AudioDecoder
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.datasets.batching import Batching, LengthBatching
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.error import DatasetError
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.datasets.utils import DynamicBatcher
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.typing import DataType


class SpeechDataset(ABC):
    """Represents a speech dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        gang: Gang,
        max_audio_len: int,
        batching: Batching,
        *,
        dtype: DataType = torch.float32,
        min_audio_len: int = 1,
        normalize_audio: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        drop_remainder: bool = False,
        sync_batches: bool = True,
        max_num_batches: int | None = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataReader[SequenceBatch]:
        """Create a dataset reader.

        :param split:
            The split to read.
        :param gang:
            The gang over which to shard the dataset.
        :param max_audio_len:
            The maximum audio length of each example. Examples longer than
            this value will be dropped.
        :param batching:
            The batching strategy for returned examples.
        :param dtype:
            The data type of the decoded audio sequences.
        :param min_audio_len:
            The minimum audio length of each example. Examples shorter than
            this value will be dropped.
        :param normalize_audio:
            If ``True``, normalizes audio to have zero mean and unit variance.
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
        :param seed:
            The seed to initialize the random number generators used internally.
        :param extras:
            The extra parameters specific to the dataset implementation.
        """

    @abstractmethod
    def splits(self) -> set[str]:
        """Return the set of splits."""


load_speech_dataset = DelegatingDatasetLoader[SpeechDataset]()


# TODO: FIX, INFER
npc = 10


@final
class GenericSpeechDataset(SpeechDataset):
    """Represents a generic manifest-based Speech dataset."""

    _dataset_name: str
    _manifest_dir: Path
    _splits: set[str]

    def __init__(self, manifest_dir: Path, splits: set[str]) -> None:
        """
        :param dataset_name:
            The name of the dataset.
        :param manifest_dir:
            The directory under which the manifest files resides.
        """
        self._manifest_dir = manifest_dir
        self._splits = splits

    @classmethod
    def from_path(cls, path: Path) -> GenericSpeechDataset:
        """Load a :class:`GenericSpeechDataset` from ``path``."""
        path = path.expanduser().resolve()

        if not path.is_dir():
            return GenericSpeechDataset(manifest_dir=path.parent, splits={path.stem})

        try:
            splits = {f.stem for f in path.glob("*.tsv")}
        except OSError as ex:
            raise RuntimeError(
                "The splits cannot be determined. See nested exception for details."
            ) from ex

        return GenericSpeechDataset(path, splits)

    @override
    def create_reader(
        self,
        split: str,
        gang: Gang,
        max_audio_len: int,
        batching: Batching,
        *,
        dtype: DataType = torch.float32,
        min_audio_len: int = 1,
        normalize_audio: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        drop_remainder: bool = False,
        sync_batches: bool = True,
        max_num_batches: int | None = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        cached_fd_count: int = 1000,
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        """
        :param cached_fd_count:
            The maximum number of file descriptors to keep open while reading
            audio files.
        """
        if split not in self._splits:
            raise ValueError(
                f"`split` must be one of the following splits, but is '{split}' instead: {', '.join(sorted(self._splits))}"
            )

        audio_dir = self._retrieve_data_directory(split)

        builder = self._builder_from_manifest(split, min_audio_len)

        def reorder_manifest(
            builder: DataPipelineBuilder,
        ) -> DataPipelineBuilder:
            manifest = list(builder.and_return())
            sizes = np.array([sample["audio_size"] for sample in manifest])
            random_order = np.random.permutation(len(sizes))
            capped_sizes = np.minimum(sizes, max_audio_len)
            indices = np.lexsort((random_order, capped_sizes))[::-1]
            sorted_manifest = [manifest[idx] for idx in indices] + [{"audio_size": -1}]
            return read_sequence(sorted_manifest)

        builder = reorder_manifest(builder)

        # Cap audio sizes by max_audio_len.
        builder.map(lambda x: min(max_audio_len, x), selector="audio_size")

        if isinstance(batching, LengthBatching):
            batcher = DynamicBatcher(-1, batching.max_num_elements, 8)

            builder.dynamic_bucket(
                1, batcher.cost_fn, bucket_creation_fn=batcher.bucket_creation_fn
            ).map(
                lambda bucket: bucket[:-1]
                if len(bucket) > 0 and bucket[-1]["audio_size"] == -1
                else bucket
            ).filter(
                lambda bucket: len(bucket) > 0
            )
        else:
            raise RuntimeError(f"`{batching}` is not supported.")

        # Shuffle buckets.
        if batch_shuffle_window != 1:
            builder.shuffle(batch_shuffle_window, seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        # Memory map audio files.
        file_mapper = FileMapper(audio_dir, cached_fd_count=cached_fd_count)

        builder.map(file_mapper, selector="[*].audio")

        # Decode audio.
        audio_decoder = AudioDecoder(dtype=torch.float32 if normalize_audio else dtype)

        builder.map(audio_decoder, selector="[*].audio.data")

        # Normalize audio if requested.
        def normalize(waveform: Tensor) -> Tensor:
            with torch.no_grad():
                waveform = layer_norm(waveform, waveform.shape)

            return waveform.to(dtype)

        if normalize_audio:
            builder.map(normalize, selector="[*].audio.data.waveform")

        def crop_audios_in_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            min_audio_len_batch = min(
                (item["audio"]["data"]["waveform"].size(0) for item in batch)
            )
            crop_size = min(max_audio_len, min_audio_len_batch)

            def crop_audio(audio: Tensor, crop_size: int) -> Tensor:
                size = audio.size(0)
                if size > crop_size:
                    start = np.random.randint(0, size - crop_size + 1)
                    return audio[start : start + crop_size]
                return audio

            for item in batch:
                item["audio"]["data"]["waveform"] = crop_audio(
                    item["audio"]["data"]["waveform"], crop_size
                )

            return batch

        builder.map(crop_audios_in_batch)

        collater = Collater()

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(num_prefetch)

        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            seqs = example["audio"]["data"]["waveform"].to(gang.device)

            return SequenceBatch(seqs, None, example=example)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=drop_remainder,
            sync_batches=sync_batches,
        )

    def _retrieve_data_directory(self, split: str) -> Path:
        manifest_file = self._manifest_dir.joinpath(f"{split}.tsv")

        try:
            with manifest_file.open() as fp:
                line = fp.readline().rstrip()
        except OSError as ex:
            raise DatasetError(
                f"{manifest_file} cannot be read. See nested exception for details."
            ) from ex

        try:
            return Path(line)
        except ValueError:
            raise DatasetError(
                f"The first line of {manifest_file} must point to a data directory."
            ) from None

    def _builder_from_manifest(
        self, split: str, min_audio_len: int
    ) -> DataPipelineBuilder:
        tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")

        builder = read_text(tsv_file, rtrim=True, memory_map=True)

        builder.skip(1)  # Path to the data directory.

        field_splitter = StrSplitter(names=["audio", "audio_size"])

        builder.map(field_splitter, num_parallel_calls=npc)

        # Cast audio size to integer.
        builder.map(int, selector="audio_size")

        builder.filter(lambda sample: sample["audio_size"] >= min_audio_len)

        return builder

    @override
    def splits(self) -> set[str]:
        return self._splits


@final
class GenericSpeechDatasetLoader(AbstractDatasetLoader[GenericSpeechDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> GenericSpeechDataset:
        try:
            return GenericSpeechDataset.from_path(path)
        except RuntimeError as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex


load_generic_speech_dataset = GenericSpeechDatasetLoader()

load_speech_dataset.register("generic_speech", load_generic_speech_dataset)
