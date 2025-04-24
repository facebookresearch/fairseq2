# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Any, Dict, Final, final, List

import numpy as np
import torch

from fairseq2.data import Collater, create_bucket_sizes, DataPipelineBuilder, FileMapper
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.text import read_text, StrSplitter
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
from fairseq2.logging import log
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import DataType, Device
from torch import Tensor
from torch.nn.functional import layer_norm
from typing_extensions import override


@torch.no_grad()
def postprocess(waveform: Tensor, normalize_audio: bool, dtype: DataType) -> Tensor:
    if waveform.dim() == 2:
        # reduce channels inplace to save the memory
        size = waveform.size(1)
        result = reduce(
            torch.Tensor.add_, [waveform[:, i] for i in range(1, size)], waveform[:, 0]
        )
        waveform = result
        waveform /= size

    if normalize_audio:
        waveform = layer_norm(waveform, waveform.shape)

    return waveform.to(dtype)


class AudioCropper:

    audio_feature: str = "audio_feature"

    def __init__(
        self, max_audio_len: int, seed: int, crop_to_batch_minimal_size: bool = False
    ) -> None:
        self.rng: np.random.RandomState = np.random.RandomState(seed)
        self.max_audio_len: int = max_audio_len
        self.crop_to_batch_minimal_size: bool = crop_to_batch_minimal_size

    def crop_audios_in_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.crop_to_batch_minimal_size:
            min_audio_len_batch = min(
                (item[self.audio_feature].size(0) for item in batch)
            )
            crop_size = min(self.max_audio_len, min_audio_len_batch)
        else:
            crop_size = self.max_audio_len

        for item in batch:
            audio = item[self.audio_feature]
            audio_size = audio.size(0)
            if audio_size > crop_size:
                start = self.rng.randint(0, audio_size - crop_size + 1)
                item[self.audio_feature] = audio[start : start + crop_size]
        return batch


def rename_feature(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for example in batch:
        if "fbank" in example["audio"]["data"]:
            example["audio_feature"] = example["audio"]["data"].pop("fbank")
        elif "waveform" in example["audio"]["data"]:
            example["audio_feature"] = example["audio"]["data"].pop("waveform")
    return batch


def to_batch(
    example: dict[str, Any], no_padding: bool, device: Device
) -> SequenceBatch:
    audio_feature = example["audio_feature"]
    if no_padding:
        seqs = audio_feature.to(device)
        padding_mask = None
    else:
        seqs, padding_mask = get_seqs_and_padding_mask(audio_feature, device=device)

    return SequenceBatch(seqs, padding_mask, example=example)


@dataclass(kw_only=True)
class SpeechReadOptions(DataReadOptions):

    dtype: DataType = torch.float32
    """The data type of the decoded audio sequences."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    use_fbank: bool = False
    """If ``True``, use fbank features instead of waveform."""

    no_padding: bool = True
    """If ``True``, all elements in the batch will be truncated to by batch minimal length.
    Therefore, no padding will be applied to the batch.
    """

    npc: int = 10
    """The number of parallel calls to use in the pipeline."""


class SpeechDataset(ABC):
    """Represents a speech dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
        options: SpeechReadOptions = SpeechReadOptions(),
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
get_speech_dataset_hub = DatasetHubAccessor(SpeechDataset)


class GenericSpeechDataset(SpeechDataset):
    """Represents a generic manifest-based Speech dataset."""

    _name: str
    _manifest_dir: Path
    _splits: set[str]

    def __init__(self, name: str, manifest_dir: Path, splits: set[str]) -> None:
        """
        :param manifest_dir:
            The directory under which the manifest files resides.
        :param splits:
            The available splits.
        """
        self._name = name
        self._manifest_dir = manifest_dir
        self._splits = splits

    @staticmethod
    def from_path(path: Path, name: str) -> GenericSpeechDataset:
        path = path.expanduser().resolve()

        if not path.is_dir():
            return GenericSpeechDataset(
                name, manifest_dir=path.parent, splits={path.stem}
            )

        try:
            splits = {f.stem for f in path.glob("*.tsv")}
        except OSError as ex:
            raise DatasetLoadError(
                name, f"The splits under the '{path}' directory of the '{name}' dataset cannot be determined. See the nested exception for details."  # fmt: skip
            ) from ex

        return GenericSpeechDataset(name, path, splits)

    @override
    def splits(self) -> set[str]:
        return self._splits

    def add_audio_reading_pipeline(
        self,
        builder: DataPipelineBuilder,
        audio_dir: Path,
        options: SpeechReadOptions,
        seed: int,
        max_audio_len: int,
    ) -> DataPipelineBuilder:
        # Memory map audio files.
        cached_fd_count = options.extras.get("cached_fd_count", 100)
        if not isinstance(cached_fd_count, int):
            raise TypeError(
                f"`options.extras['cached_fd_count']` must be of type `int`, but is of type `{type(cached_fd_count)}` instead."
            )

        file_mapper = FileMapper(audio_dir, cached_fd_count=cached_fd_count)

        builder.map(file_mapper, selector="[*].audio")

        # Decode audio.
        audio_decoder = AudioDecoder(
            dtype=torch.float32 if options.normalize_audio else options.dtype
        )
        builder.map(
            audio_decoder, selector="[*].audio.data", num_parallel_calls=options.npc
        )

        if options.use_fbank:
            fbank_converter = WaveformToFbankConverter(
                num_mel_bins=80,
                waveform_scale=2**15,
                channel_last=True,
                standardize=True,
                dtype=options.dtype,
            )

            builder.map(
                fbank_converter,
                selector="[*].audio.data",
                num_parallel_calls=options.npc,
            )
        else:
            builder.map(
                partial(
                    postprocess,
                    normalize_audio=options.normalize_audio,
                    dtype=options.dtype,
                ),
                selector="[*].audio.data.waveform",
            )

        # select the audio feature at the top level
        builder.map(rename_feature)

        # Crop long audios to `max_audio_len`.
        audio_cropper = AudioCropper(
            max_audio_len,
            seed=seed,
            crop_to_batch_minimal_size=options.no_padding,
        )
        builder.map(audio_cropper.crop_audios_in_batch)

        return builder

    def _retrieve_data_directory(self, split: str) -> Path:
        manifest_file = self._manifest_dir.joinpath(f"{split}.tsv")

        try:
            with manifest_file.open(encoding="utf-8") as fp:
                line = fp.readline().rstrip()
        except OSError as ex:
            raise DataReadError(
                self._name, split, f"The {manifest_file} manifest file cannot be read. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            return Path(line)
        except ValueError:
            raise DataReadError(
                self._name, split, f"The first line of the '{manifest_file}' manifest file must point to a data directory."  # fmt: skip
            ) from None

    def _read_manifest(
        self, split: str, max_audio_len: int, min_audio_len: int, audio_dir: Path | None
    ) -> DataPipelineBuilder:
        """
        we only apply min_audio_len filter here,
        longer audio will be croped to max_audio_len latter
        """
        tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")

        builder = read_text(
            tsv_file, rtrim=True, memory_map=True, block_size=10 * 1024 * 1024
        )

        if audio_dir is not None:
            builder.skip(1)  # Path to the data directory.

        field_splitter = StrSplitter(names=["audio", "audio_size"])

        builder.map(field_splitter)

        builder.map(
            lambda x: min(int(x), max_audio_len),
            selector="audio_size",
        )

        builder.filter(lambda sample: sample["audio_size"] >= min_audio_len)

        return builder

    def add_bucketing_pipeline(
        self,
        builder: DataPipelineBuilder,
        options: SpeechReadOptions,
        max_audio_len: int,
        min_audio_len: int,
        seed: int,
        columns: str,
    ) -> DataPipelineBuilder:
        batching = options.batching

        if isinstance(batching, LengthBatching):
            # Bucket by the audio length.
            max_num_elements = batching.max_num_elements
            num_seqs_multiple_of = options.extras.get("num_seqs_multiple_of", 8)
            assert isinstance(
                num_seqs_multiple_of, int
            ), "num_seqs_multiple_of must be an integer"
            assert num_seqs_multiple_of > 0, "num_seqs_multiple_of must be positive"

            if max_num_elements % max_audio_len != 0:
                max_num_elements = (max_num_elements // max_audio_len) * max_audio_len
                log.warning(f"`max_num_elements` is rounded to {max_num_elements}")

            bucket_sizes = create_bucket_sizes(
                min_seq_len=min_audio_len,
                max_seq_len=max_audio_len,
                max_num_elements=max_num_elements,
                num_seqs_multiple_of=num_seqs_multiple_of,
            )

            builder.bucket_by_length(
                bucket_sizes,
                selector=columns,
                min_data_len=min_audio_len,
                skip_below_min_examples=True,  # this should be neutral
                skip_above_max_examples=True,  # this should be neutral
                drop_remainder=options.drop_remainder,
            )
        elif isinstance(batching, StaticBatching):
            if options.no_padding:
                raise NotSupportedError(
                    "no_padding is not supported for static batching"
                )
            builder.bucket(batching.batch_size, drop_remainder=options.drop_remainder)
        else:
            raise NotSupportedError(f"`{batching}` is not supported.")

        # Shuffle buckets.
        if options.batch_shuffle_window != 1:
            builder.shuffle(options.batch_shuffle_window, seed)
        return builder

    @override
    def create_pipeline(
        self,
        split: str,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
        options: SpeechReadOptions | None = None,
    ) -> DataPipelineReader[SequenceBatch]:

        if split not in self._splits:
            raise UnknownSplitError(self._name, split, self._splits)

        if options is None:
            options = SpeechReadOptions()

        log.info(
            f"Creating a reader for the <{split}> split of the <{self._name}>"
            f" dataset with the following options:/n {options}."
        )

        seed = options.seed
        npc = options.npc
        no_padding = options.no_padding

        audio_dir = self._retrieve_data_directory(split)
        builder = self._read_manifest(split, max_audio_len, min_audio_len, audio_dir)

        if options.example_shuffle_window != 1:
            # builder.prefetch(options.example_shuffle_window * options.num_prefetch)
            builder.shuffle(options.example_shuffle_window, seed)
            seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)
        seed += gang.rank

        builder = self.add_bucketing_pipeline(
            builder, options, max_audio_len, min_audio_len, seed, "audio_size"
        )
        seed += 1
        builder = self.add_audio_reading_pipeline(
            builder, audio_dir, options, seed, max_audio_len
        )
        # Collate batched examples into a batch.
        collater = Collater(pad_value=None if no_padding else 0)
        builder.map(collater)

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        builder.prefetch(options.num_prefetch)

        pipeline = builder.map(
            partial(to_batch, no_padding=no_padding, device=gang.device)
        )
        return pipeline

    @override
    def create_reader(
        self,
        split: str,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
        options: SpeechReadOptions | None = None,
    ) -> DataPipelineReader[SequenceBatch]:
        pipeline = self.create_pipeline(
            split, gang, min_audio_len, max_audio_len, options
        )
        return DataPipelineReader[SequenceBatch](
            self._name, split, pipeline, gang, options
        )
