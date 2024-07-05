# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Set, final

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import layer_norm

from fairseq2.assets import AssetCard
from fairseq2.data import (
    Collater,
    DataPipelineBuilder,
    FileMapper,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.data.audio import AudioDecoder
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.error import DatasetError
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.gang import Gang
from fairseq2.typing import DataType, override


class SpeechDataset(ABC):
    """Represents an speech dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        gang: Gang,
        max_audio_len: int,
        max_num_elements: int,
        *,
        dtype: DataType = torch.float32,
        min_audio_len: int = 1,
        normalize_audio: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataReader[Tensor]:
        """Create a dataset reader.

        :param split:
            The split to read.
        :param gang:
            The gang over which to shard the dataset.
        :param max_audio_len:
            The maximum audio length of each example. Examples longer than
            this value will be cropped.
        :param max_num_elements:
            The maximum number of elements in each batch.
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


load_speech_dataset = DelegatingDatasetLoader[SpeechDataset]()


# TODO: FIX, INFER
npc = 10


# TODO: Work in progress!
@final
class GenericSpeechDataset(SpeechDataset):
    """Represents a generic manifest-based Speech dataset."""

    _dataset_name: str
    _manifest_dir: Path
    _splits: Set[str]

    def __init__(self, dataset_name: str, manifest_dir: Path) -> None:
        """
        :param dataset_name:
            The name of the dataset.
        :param manifest_dir:
            The directory under which the manifest files resides.
        """
        self._dataset_name = dataset_name
        self._manifest_dir = manifest_dir

        self._splits = set()

        for tsv_file in manifest_dir.glob("*.tsv"):
            self._splits.add(tsv_file.stem)

    @override
    def create_reader(
        self,
        split: str,
        gang: Gang,
        max_audio_len: int,
        max_num_elements: int,
        *,
        dtype: DataType = torch.float32,
        min_audio_len: int = 1,
        normalize_audio: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        cached_fd_count: int = 1000,
        **extras: Any,
    ) -> DataPipelineReader[Tensor]:
        """
        :param cached_fd_count:
            The maximum number of file descriptors to keep open while reading
            audio files.
        """
        if split not in self._splits:
            raise ValueError(
                f"`split` must be a valid split name, but the {self._dataset_name} dataset has no split named '{split}'."
            )

        root_data_dir = self._retrieve_data_directory(split)

        builder = self._builder_from_manifest(split, max_audio_len)

        # TODO: Remove this hack which is done just to get parity with fairseq1's dataloader.
        def fairseq1_hack(builder: DataPipelineBuilder) -> DataPipelineBuilder:
            manifest = list(builder.and_return())
            manifest = list(
                filter(
                    lambda sample: int(sample["audio_size"]) >= min_audio_len, manifest
                )
            )
            sizes = np.array([int(sample["audio_size"]) for sample in manifest])
            random_order = np.random.permutation(len(sizes))
            limited_sizes = np.minimum(sizes, max_audio_len)
            indices = np.lexsort((random_order, limited_sizes))[::-1]
            sorted_manifest = [manifest[idx] for idx in indices]
            return read_sequence(sorted_manifest)

        builder = fairseq1_hack(builder)

        # TODO: Add this back once we remove fairseq1_hack.
        # # Shuffle examples. Must be consistent across all processes.
        # if example_shuffle_window != 1:
        #     builder.shuffle(example_shuffle_window, seed)

        # seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        # Bucket by audio length.
        bucket_sizes = create_bucket_sizes(
            max_num_elements=max_num_elements,
            max_seq_len=max_audio_len,
            min_seq_len=min_audio_len,
            num_seqs_multiple_of=8,
        )

        builder.bucket_by_length(
            bucket_sizes,
            selector="audio_size",
            min_data_len=min_audio_len,
            skip_below_min_examples=True,
            skip_above_max_examples=True,
        )

        # Shuffle buckets.
        if batch_shuffle_window != 1:
            builder.shuffle(batch_shuffle_window, seed)

        seed += 1

        # Memory map audio files.
        file_mapper = FileMapper(root_data_dir, cached_fd_count=cached_fd_count)

        builder.map(file_mapper, selector="[*].audio")

        # Decode audio.
        audio_decoder = AudioDecoder(dtype=torch.float32 if normalize_audio else dtype)

        builder.map(audio_decoder, selector="[*].audio.data")

        # TODO(balioglu): Check/adjust sample size

        # Normalize audio if requested.
        def normalize(waveform: Tensor) -> Tensor:
            with torch.no_grad():
                waveform = layer_norm(waveform, waveform.shape)

            return waveform.to(dtype)

        if normalize_audio:
            builder.map(normalize, selector="[*].audio.data.waveform")

        def crop_audios_in_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

        # Prefetch `num_prefetch` examples in background.
        builder.prefetch(num_prefetch)

        def example_to_batch(example: Dict[str, Any]) -> Tensor:
            seqs = example["audio"]["data"]["waveform"]
            if gang.device is not None:
                seqs = seqs.to(gang.device)
            return seqs

        pipeline = builder.map(example_to_batch).and_return()

        return DataPipelineReader[Tensor](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=True,
        )

    def _retrieve_data_directory(self, split: str) -> Path:
        tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")

        try:
            with tsv_file.open() as fp:
                line = fp.readline().rstrip()
        except OSError as ex:
            raise DatasetError(
                f"The manifest file '{tsv_file}' of the {self._dataset_name} dataset cannot be read. See nested exception for details."
            ) from ex

        try:
            return Path(line)
        except ValueError:
            raise DatasetError(
                f"The first line of the manifest file '{tsv_file}' of the {self._dataset_name} dataset must point to a data directory."
            )

    def _builder_from_manifest(
        self, split: str, max_audio_len: int
    ) -> DataPipelineBuilder:
        tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")

        builder = read_text(tsv_file, rtrim=True, memory_map=True)

        builder.skip(1)

        field_splitter = StrSplitter(names=["audio", "audio_size"])

        builder.map(field_splitter, num_parallel_calls=npc)

        # Manually change "audio_size" to audios longer than max_audio_len to max_audio_len.
        # This is done so that we can create buckets without error. We will anyway crop these audios.
        def cap_audio_size(audio_size: str) -> int:
            return min(max_audio_len, int(audio_size))

        builder.map(cap_audio_size, selector="audio_size")

        return builder

    @override
    def splits(self) -> Set[str]:
        return self._splits


@final
class GenericSpeechDatasetLoader(AbstractDatasetLoader[GenericSpeechDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> GenericSpeechDataset:
        return GenericSpeechDataset(card.name, path)


load_generic_speech_dataset = GenericSpeechDatasetLoader()

load_speech_dataset.register("generic_speech", load_generic_speech_dataset)
