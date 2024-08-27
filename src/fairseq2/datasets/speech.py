# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final

import torch
from typing_extensions import override

from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DataReadOptions,
    DatasetHubAccessor,
)
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.typing import DataType


@dataclass(kw_only=True)
class SpeechReadOptions(DataReadOptions):
    dtype: DataType = torch.float32
    """The data type of the decoded audio sequences."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""


class SpeechDataset(ABC):
    """Represents a speech dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
        options: SpeechReadOptions | None = None,
    ) -> DataReader[SequenceBatch]:
        """Create a dataset reader.

        :param split:
            The split to read.
        :param gang:
            The gang over which to shard the dataset.
        :param min_audio_len:
            The minimum audio length of each example. Examples shorter than this
            value will be dropped.
        :param max_audio_len:
            The maximum audio length of each example. Examples longer than this
            value will be dropped.
        :param options:
            The read options.
        """

    @abstractmethod
    def splits(self) -> set[str]:
        """Return the set of splits."""


GENERIC_SPEECH_DATASET_FAMILY: Final = "generic_speech"


class AudioCropper:
    def __init__(self, max_audio_len: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.max_audio_len = max_audio_len

    def crop_audio(self, audio: Tensor, crop_size: int) -> Tensor:
        size = audio.size(0)
        if size > crop_size:
            start = self.rng.integers(0, size - crop_size + 1)
            return audio[start : start + crop_size]
        return audio

    def crop_audios_in_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        min_audio_len_batch = min(
            (item["audio"]["data"]["waveform"].size(0) for item in batch)
        )
        crop_size = min(self.max_audio_len, min_audio_len_batch)
        for item in batch:
            item["audio"]["data"]["waveform"] = self.crop_audio(
                item["audio"]["data"]["waveform"], crop_size
            )
        return batch


class AudioCropper:
    def __init__(self, max_audio_len: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.max_audio_len = max_audio_len

    def crop_audio(self, audio: Tensor, crop_size: int) -> Tensor:
        size = audio.size(0)
        if size > crop_size:
            start = self.rng.integers(0, size - crop_size + 1)
            return audio[start : start + crop_size]
        return audio

    def crop_audios_in_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        min_audio_len_batch = min(
            (item["audio"]["data"]["waveform"].size(0) for item in batch)
        )
        crop_size = min(self.max_audio_len, min_audio_len_batch)
        for item in batch:
            item["audio"]["data"]["waveform"] = self.crop_audio(
                item["audio"]["data"]["waveform"], crop_size
            )
        return batch


@final
class GenericSpeechDataset(SpeechDataset):
    """Represents a generic manifest-based Speech dataset."""

    @staticmethod
    def from_path(path: Path, name: str) -> GenericSpeechDataset:
        return GenericSpeechDataset()

    @override
    def create_reader(
        self,
        split: str,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
        options: SpeechReadOptions | None = None,
    ) -> DataPipelineReader[SequenceBatch]:
<<<<<<< HEAD
        raise NotSupportedError("not supported yet.")
=======
        """
        :param cached_fd_count:
            The maximum number of file descriptors to keep open while reading
            audio files.
        """
        if split not in self._splits:
            raise ValueError(
                f"`split` must be one of the following splits, but is '{split}' instead: {', '.join(sorted(self._splits))}"
            )

        rng = np.random.default_rng(seed)

        seed += 1

        audio_dir = self._retrieve_data_directory(split)

        builder = self._builder_from_manifest(split, min_audio_len)

        def reorder_manifest(
            builder: DataPipelineBuilder,
            rng: np.random.Generator,
        ) -> DataPipelineBuilder:
            manifest = list(builder.and_return())
            sizes = np.array([sample["audio_size"] for sample in manifest])
            random_order = rng.permutation(len(sizes))
            capped_sizes = np.minimum(sizes, max_audio_len)
            indices = np.lexsort((random_order, capped_sizes))[::-1]
            sorted_manifest = [manifest[idx] for idx in indices] + [{"audio_size": -1}]
            return read_sequence(sorted_manifest)

        builder = reorder_manifest(builder, rng)

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

        rng = np.random.default_rng(seed)

        seed += 1

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

        audio_cropper = AudioCropper(max_audio_len, rng)

        builder.map(audio_cropper.crop_audios_in_batch)

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
>>>>>>> 7de8efcf (Fix bug in np random seed during dataloading.)

    @override
    def splits(self) -> set[str]:
        return set()


get_speech_dataset_hub = DatasetHubAccessor(SpeechDataset)
