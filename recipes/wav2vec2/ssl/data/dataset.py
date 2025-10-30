# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Final, final

import torch

from fairseq2.data.data_pipeline import Collater, DataPipelineBuilder
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.data_type import DataType
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DataReadError,
    DatasetError,
    SequenceBatch,
    SyncMode,
)
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.recipe.config import DatasetSection

from .batch_utils import add_batch_shuffling, add_length_batching
from .preprocessing import (
    add_audio_cropping,
    add_audio_decoding,
    add_audio_file_loading,
)

WAV2VEC2_SSL_DATASET: Final = "wav2vec2_ssl"


@dataclass(kw_only=True)
class Wav2Vec2SslDatasetSection(DatasetSection):
    """wav2vec2 training dataset configuration."""

    name: str | None = "librispeech_960h"
    """The name or path to the asset card of the dataset."""

    family: str = WAV2VEC2_SSL_DATASET  # type: ignore

    train_split: str | None = "train"
    """The name of the training data split. Expecting a {train_split}.tsvfile in the dataset directory. Only `None` during evaluation.
    """

    valid_split: str | None = "valid"
    """The name of the validation data split(s). Expecting a {valid_split}.tsv under `path`. Format multiple splits interspersed by "," and without spaces (`'valid,test_dev,test_clean'`).
    """

    min_audio_len: int = 32_000
    """The minimum audio sequence length."""

    max_audio_len: int = 250_000
    """The maximum audio sequence length."""

    dtype: DataType = torch.float16
    """Numerical precision for audio decoding. Overridden to `torch.float32` when ``config.normalize_audio = True``."""

    # Batching configuration
    num_seqs_multiple_of: int = 8
    """Each batch will have `<= max_num_elements` elements with `<= num_seqs_multipe_of` or `num_seqs_multiple_of * N` sequences.
    This is primarily for hardware optimization, but will only work when sufficient enough sequences are available for bucketing.
    """

    max_num_elements: int = 1_500_000
    """Each batch will have `<= max_num_elements` elements with `<= num_seqs_multipe_of` or `num_seqs_multiple_of * N` sequences.
    This is primarily for hardware optimization, but will only work when sufficient enough sequences are available for bucketing.
    """

    drop_remainder: bool = False
    """If ``True``, drops the last set of batches if they have in total fewer examples than requested."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance and uses torch.float32 during audio decoding (overriding ``config.dtype``)."""

    npc: int = 10
    """The number of parallel calls to use in the pipeline."""

    # Shuffling
    example_shuffle_window: int = 500_000
    """The size of the sliding window for shuffling examples (pre-batch). Zero shuffles the entire dataset."""

    batch_shuffle_window: int = 0
    """The size of the sliding window for shuffling batches. Zero shuffles all batches."""

    # Misc
    max_num_batches: int | None = None
    """The maximum number of batches to return."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    cached_fd_count: int = 1000
    """Enables an LRU cache on the last ``cached_fd_count`` files read.
    ``FileMapper`` will memory map all the cached file, so this is especially
    useful for reading several slices of the same file.
    """


@final
class Wav2Vec2SslDataset:
    """wav2vec2 training dataset with audio decoding support."""

    _manifest_dir: Path
    _splits: set[str]

    def __init__(self, manifest_dir: Path, splits: set[str]) -> None:
        self._manifest_dir = manifest_dir
        self._splits = splits

    @classmethod
    def from_path(cls, path: Path) -> "Wav2Vec2SslDataset":
        """Create dataset from path (file or directory)."""
        path = path.expanduser().resolve()

        if not path.is_dir():
            return cls(manifest_dir=path.parent, splits={path.stem})

        try:
            splits = {f.stem for f in path.glob("*.tsv")}
        except OSError as ex:
            raise DatasetError(
                f"The splits under the '{path}' directory of the dataset cannot be determined. See the nested exception for details.",
            ) from ex

        if not splits:
            raise DatasetError(f"No .tsv files found in {path}")

        return cls(manifest_dir=path, splits=splits)

    def _retrieve_audio_directory(self, split: str) -> Path:
        """
        Retrieve audio directory from manifest file header.
        Expecting the following structure:

        ```text (train-clean-100.tsv)
        /path-to-librispeech/librispeech/062419
        train-clean-100/1553/140047/1553-140047-0000.flac       180080
        train-clean-100/1553/140047/1553-140047-0001.flac       219840
        (...)
        ```
        """
        manifest_file = self._manifest_dir.joinpath(f"{split}.tsv")

        try:
            with manifest_file.open(encoding="utf-8") as fp:
                header = fp.readline().rstrip()
        except OSError as ex:
            raise DataReadError(
                f"The {manifest_file} manifest file cannot be read. See the nested exception for details.",
            ) from ex

        audio_dir = Path(header)
        if not audio_dir.is_dir():
            raise DataReadError(
                f"{audio_dir} pointed by the {manifest_file} manifest file is not a directory.",
            )

        return audio_dir

    def _read_manifest(self, split: str) -> DataPipelineBuilder:
        """
        Read and parse TSV manifest file.
        Expecting the following structure:

        ```text (train-clean-100.tsv)
        /path-to-librispeech/librispeech/062419
        train-clean-100/1553/140047/1553-140047-0000.flac       180080
        train-clean-100/1553/140047/1553-140047-0001.flac       219840
        (...)
        ```
        """
        tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")

        builder = read_text(
            tsv_file,
            rtrim=True,
            memory_map=True,
        )

        builder.skip(1)  # Skip one line to the path to the data directory

        field_splitter = StrSplitter(names=["audio", "audio_size"])
        builder.map(field_splitter)

        # Convert audio_size to int
        return builder.map(int, selector="audio_size")

    @staticmethod
    def create_sequence_batch(batch_dict: dict[str, Any]) -> SequenceBatch:
        feature_tensor = batch_dict["audio"]["data"]["waveform"]
        return SequenceBatch(feature_tensor, seq_lens=None, example=batch_dict)

    def create_reader(
        self,
        split: str,
        gangs: Gangs,
        min_audio_len: int,
        max_audio_len: int,
        *,
        # Batching
        max_num_elements: int,
        num_seqs_multiple_of: int,
        # Audio processing
        dtype: torch.dtype,
        normalize_audio: bool,
        npc: int,
        # Shuffling
        example_shuffle_window: int,
        batch_shuffle_window: int,
        # Misc
        num_prefetch: int,
        drop_remainder: bool,
        sync_mode: SyncMode,
        seed: int,
        max_num_batches: int | None,
        cached_fd_count: int,
        # Provided by TrainerSection
        num_accumulate: int,
    ) -> DataReader[SequenceBatch]:
        """Create data reader with complete audio processing pipeline."""

        if split not in self._splits:
            raise DataReadError(f"Unknown split '{split}'. Available: {self._splits}")

        log.info(f"Creating a reader for the <{split}> split of the dataset.")

        # Read audio directory path from the first line of the manifest
        audio_dir = self._retrieve_audio_directory(split)

        # Read TSV manifest -> { audio: str, audio_size: int }
        builder = self._read_manifest(split)

        # Shuffle the dataset samples
        if example_shuffle_window != 1:
            builder.shuffle(example_shuffle_window, seed)
            seed += 1

        # Shard across distributed processes
        if gangs.dp.size > 1:
            builder.shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)
            seed += gangs.dp.rank

        # Length batching -> list( { audio: str, audio_size: int } )
        builder = add_length_batching(
            builder,
            min_audio_len=min_audio_len,
            max_audio_len=max_audio_len,
            max_num_elements=max_num_elements,
            num_seqs_multiple_of=num_seqs_multiple_of,
            drop_remainder=drop_remainder,
            selector="audio_size",
        )
        # Batch shuffling
        builder = add_batch_shuffling(
            builder, batch_shuffle_window=batch_shuffle_window, seed=seed
        )
        seed += 1

        # Path resolution -> list( { audio.path: str, audio_size: int } )
        builder = add_audio_file_loading(
            builder,
            audio_dir=audio_dir,
            cached_fd_count=cached_fd_count,
            selector="[*].audio",
        )

        # Audio decoding -> list ( { audio.path: str, audio.data.sample_rate: int, audio.data.format: int
        #                            audio.data.waveform: tensor, audio.audio_size: int } )
        builder = add_audio_decoding(
            builder,
            dtype=dtype,
            normalize_audio=normalize_audio,
            npc=npc,
            selector="[*].audio.data",
        )

        # Audio cropping
        builder = add_audio_cropping(
            builder,
            seed=seed,
            max_audio_len=max_audio_len,
            crop_to_batch_minimal_size=True,
            audio_feature_selector="audio.data.waveform",
        )

        # Collation -> { audio.path: list(str), audio.data.sample_rate: list(int), audio.data.format: list(int),
        #                audio_size: list(int), audio.data.waveform: list(tensor) }
        collater = Collater(pad_value=None)
        builder.map(collater)

        # Limit batches
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch
        builder.prefetch(num_prefetch)

        # Wrap in SequenceBatch
        builder.map(partial(Wav2Vec2SslDataset.create_sequence_batch))

        pipeline = builder.and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline,
            gangs,
            num_accumulate=num_accumulate,
            sync=True,
            sync_mode=sync_mode,
        )


@dataclass
class Wav2Vec2SslDatasetConfig:
    """
    This configuration matches the keys after the top-level `dataset_config:` key
    in the YAML asset definition:

    ```yaml
    name: mydataset
    dataset_config:
      manifest_dir:
    ```
    """

    manifest_dir: Path = field(default_factory=Path)


def open_wav2vec2_ssl_dataset(config: Wav2Vec2SslDatasetConfig) -> Wav2Vec2SslDataset:
    """The mapping between the dataset asset card definition and the Wav2Vec2SslDataset."""
    return Wav2Vec2SslDataset.from_path(config.manifest_dir)
