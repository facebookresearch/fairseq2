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
from typing import Any, Dict, Final, List, final

from fairseq2.datasets.speech import SpeechDataset, SpeechReadOptions
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import layer_norm
from typing_extensions import override

from fairseq2.data import (
    Collater,
    DataPipelineBuilder,
    FileMapper,
    MemoryBlock,
    create_bucket_sizes,
)

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from fairseq2.data.parquet import *

from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.text import StrSplitter, read_text
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
from fairseq2.typing import DataType


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




PARQUET_SPEECH_DATASET_FAMILY: Final = "generic_parquet_speech"


@final
class GenericParquetSpeechDataset(SpeechDataset):
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

    @override
    def create_reader(
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
            builder.prefetch(options.example_shuffle_window * options.num_prefetch)
            builder.shuffle(options.example_shuffle_window, seed)
            seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)
        seed += gang.rank

        batching = options.batching

        if isinstance(batching, LengthBatching):
            # Bucket by the audio length.
            max_num_elements = batching.max_num_elements
            if max_num_elements % max_audio_len != 0:
                max_num_elements = (
                    max_num_elements // max_audio_len + 1
                ) * max_audio_len
                log.warning(f"`max_num_elements` is rounded to {max_num_elements}")

            bucket_sizes = create_bucket_sizes(
                min_seq_len=min_audio_len,
                max_seq_len=max_audio_len,
                max_num_elements=max_num_elements,
                num_seqs_multiple_of=8,
            )

            builder.bucket_by_length(
                bucket_sizes,
                selector="audio_size",
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

        seed += 1

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
        builder.map(audio_decoder, selector="[*].audio.data", num_parallel_calls=npc)

        if options.use_fbank:
            fbank_converter = WaveformToFbankConverter(
                num_mel_bins=80,
                waveform_scale=2**15,
                channel_last=True,
                standardize=True,
                dtype=options.dtype,
            )

            builder.map(
                fbank_converter, selector="[*].audio.data", num_parallel_calls=npc
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
            crop_to_batch_minimal_size=no_padding,
        )
        builder.map(audio_cropper.crop_audios_in_batch)

        # Collate batched examples into a batch.
        pad_value = None if no_padding else 0
        collater = Collater(pad_value=pad_value)
        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        builder.prefetch(options.num_prefetch)

        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            audio_feature = example["audio_feature"]
            if no_padding:
                seqs = audio_feature.to(gang.device)
                padding_mask = None
            else:
                seqs, padding_mask = get_seqs_and_padding_mask(
                    audio_feature, device=gang.device
                )

            return SequenceBatch(seqs, padding_mask, example=example)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            self._name, split, pipeline, gang, options
        )


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
from fairseq2.assets import AssetCard, AssetError
from fairseq2.data import Collater, DataPipelineBuilder, FileMapper, MemoryBlock, read_sequence
from fairseq2.data.audio import AudioDecoder
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.datasets.batching import Batching, LengthBatching
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.error import DatasetError
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models.sequence import SequenceBatch
from fairseq2.typing import DataType
from torch import Tensor
from torch.nn.functional import layer_norm
from typing_extensions import override

from mms.datasets.utils import batch_by_size_vec
from src.mms.datasets.speech import AudioCropper, SpeechDataset

import os
os.environ['OMP_NUM_THREADS'] = '1'

# os.environ['MKL_NUM_THREADS'] = '4'
# os.environ['OPENBLAS_NUM_THREADS'] = '4'


from dataclasses import dataclass
from functools import reduce
from typing import Any, List
from fairseq2.data import MemoryBlock, create_bucket_sizes, read_sequence
from fairseq2.data.audio import AudioDecoder
import pyarrow as pa
import polars as pl
import pyarrow.compute as pc
from fairseq2.data.parquet import *
import numpy as np
from torch.nn.functional import layer_norm


log = get_log_writer(__name__)


load_speech_dataset = DelegatingDatasetLoader[SpeechDataset]()


import pyarrow.parquet as pq
import pyarrow as pa


@final
class GenericSpeechParquetDataset(SpeechDataset):
    """Represents a generic manifest-based Speech dataset."""

    _dataset: pq.ParquetDataset
    _splits: set[str]
    split_column: str = "split"

    def __init__(self, dataset: pq.ParquetDataset, splits: set[str]) -> None:
        self._dataset = dataset
        self._splits = splits

    @override
    def splits(self) -> set[str]:
        return self._splits

    @classmethod
    def from_path(cls, path: Path | str) -> GenericSpeechParquetDataset:
        from stopes.fb_config import get_filesystem_from_path  # to work with BlobStore filesystem
        fixed_path, filesystem = get_filesystem_from_path(path)
        datasest = pq.ParquetDataset(fixed_path, filesystem=filesystem)

        partition_columns = datasest.partitioning.schema.names

        if cls.split_column in partition_columns:
            idx = partition_columns.index(cls.split_column)
            _splits = datasest.partitioning.dictionaries[idx]
            if _splits is None:
                splits = set()
            else:
                splits = set(_splits.to_pylist())
        else:
            splits = set()

        return GenericSpeechParquetDataset(datasest, splits)

    @override
    def create_reader(
        self,
        split: str | None,
        gang: Gang,
        max_audio_len: int,
        batching: Batching,
        *,
        dtype: DataType = torch.float32,
        min_audio_len: int = 1,
        normalize_audio: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        is_binarized: bool = False,
        drop_remainder: bool = False,
        sync_batches: bool = False,
        max_num_batches: int | None = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        bsz_mult: int = 8,
        npc: int = 12,
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        if split is not None and split not in self._splits:
            raise ValueError(
                f"`split` must be one of the following splits, but is '{split}' instead: {', '.join(sorted(self._splits))}"
            )

        audio_decoder = AudioDecoder(dtype=torch.float32 if normalize_audio else dtype)
        def decoded_audio(_bytes):
            return audio_decoder(MemoryBlock(_bytes.tobytes()))


        if isinstance(batching, LengthBatching):
            indices = np.ascontiguousarray(indices, dtype=np.uint32)
            num_tokens_vec = np.take(sizes, indices)
            batches = batch_by_size_vec(
                indices,
                num_tokens_vec,
                batching.max_num_elements,
                -1,
                bsz_mult,
            )
            log.info("Created {} batches.", split)
        else:
            raise RuntimeError(f"`{batching}` is not supported.")

        # Shuffle buckets.
        if batch_shuffle_window != 1:
            rng = np.random.default_rng(seed)
            seed += 1
            rng.shuffle(batches)
            log.info("Shuffled {} batches.", len(batches))

        builder = read_sequence(batches)

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=False)

        seed += gang.rank

        if is_binarized:

            def get_paths(batch: list[np.ndarray]) -> list[dict[str, str]]:
                batched_paths = []
                for index in batch:
                    file_path_tokens = file_paths[index].tolist()
                    file_path = "".join(
                        [
                            symbols[token]
                            for token in file_path_tokens
                            if token not in (0, 2)
                        ]
                    )
                    batched_paths.append({"audio": file_path})
                return batched_paths

            builder.map(get_paths)
        else:
            builder.map(
                lambda batch: [{"audio": file_paths[index]} for index in batch],
            )

        # Memory map audio files.
        file_mapper = FileMapper(audio_dir, cached_fd_count=cached_fd_count)

        builder.map(file_mapper, selector="[*].audio", num_parallel_calls=npc)

        # Decode audio.
        audio_decoder = AudioDecoder(dtype=torch.float32 if normalize_audio else dtype)

        builder.map(audio_decoder, selector="[*].audio.data", num_parallel_calls=npc)

        # Normalize audio if requested.
        def postprocess(waveform: Tensor) -> Tensor:
            if waveform.dim() == 2:
                waveform = waveform.mean(-1)

            if normalize_audio:
                with torch.no_grad():
                    waveform = layer_norm(waveform, waveform.shape)

            return waveform.to(dtype)

        builder.map(
            postprocess,
            selector="[*].audio.data.waveform",
        )

        rng = np.random.default_rng(seed)
        seed += 1

        audio_cropper = AudioCropper(max_audio_len, rng)

        builder.map(audio_cropper.crop_audios_in_batch)

        collater = Collater()

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        if num_prefetch > 0:
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
