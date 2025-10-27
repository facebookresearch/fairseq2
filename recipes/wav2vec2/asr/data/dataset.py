# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, final

import torch

from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.data_type import DataType
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DataReadError,
    DatasetError,
    Seq2SeqBatch,
    SyncMode,
)
from fairseq2.gang import Gangs
from fairseq2.recipe.config import DatasetSection

from .batch_utils import add_batch_shuffling, add_length_batching
from .preprocessing import (
    add_audio_decoding,
    add_audio_file_loading,
    add_layernorm,
    collate_with_pad_ix,
    encode_text,
    filter_by_min_max_audio_size,
)

WAV2VEC2_ASR_DATASET: Final = "wav2vec2_asr"


@dataclass(kw_only=True)
class Wav2Vec2AsrDatasetSection(DatasetSection):
    """wav2vec2 ASR dataset configuration section."""

    name: str | None = "librilight_asr_10h"
    """The name, path or path to the asset card of the speech dataset."""

    family: str = WAV2VEC2_ASR_DATASET  # type: ignore

    train_split: str | None = "train"
    """The name of the training data split. Expecting a {train_split}.tsv file in the dataset directory. Only `None` during evaluation.
    """

    valid_split: str | None = "dev_other"
    """The name of the validation data split. Expecting a {valid_split}.tsv and {valid_split}.wrd files in the dataset directory.
    """

    min_audio_len: int = 1
    """The minimum audio sequence length."""

    max_audio_len: int = 800_000
    """The maximum audio sequence length."""

    dtype: DataType = torch.float16
    """Numerical precision for audio decoding. Overridden to `torch.float32` when ``config.normalize_audio = True``."""

    # Batching parameters
    num_seqs_multiple_of: int = 8
    """Each batch will have `<= max_num_elements` elements with `<= num_seqs_multipe_of` or `num_seqs_multiple_of * N` sequences.
    This is primarily for hardware optimization, but will only work when sufficient enough sequences are available for bucketing.
    """

    max_num_elements: int = 3_200_000
    """Ignores `batch_size` and each batch will have `<= max_num_elements` elements with `<= num_seqs_multipe_of` or `num_seqs_multiple_of * N` sequences.
    """

    drop_remainder: bool = False
    """If ``True``, drops the last set of batches if they have in total fewer examples than requested."""

    # Audio processing
    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance and uses torch.float32 during audio decoding (overriding ``config.dtype``)."""

    npc: int = 10
    """The number of parallel calls to use in the pipeline."""

    # Shuffling parameters
    example_shuffle_window: int = 0
    """The size of the sliding window for shuffling examples (pre-batch)."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

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
class Wav2Vec2AsrDataset:
    """wav2vec2 ASR dataset with audio decoding and text tokenization support."""

    _manifest_dir: Path
    _splits: set[str]

    def __init__(self, manifest_dir: Path, splits: set[str]) -> None:
        self._manifest_dir = manifest_dir
        self._splits = splits

    @classmethod
    def from_path(cls, path: Path) -> "Wav2Vec2AsrDataset":
        """Create dataset from path (file or directory)."""
        path = path.expanduser().resolve()

        if not path.is_dir():
            return cls(manifest_dir=path.parent, splits={path.stem})

        try:
            splits = {f.stem for f in path.glob("*.tsv")}
        except OSError as ex:
            raise DatasetError(
                f"The splits under the '{path}' directory cannot be determined. See the nested exception for details."
            ) from ex

        if not splits:
            raise DatasetError(f"The '{path}' directory does not contain any splits.")

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

    def _read_wrd_file(self, split: str) -> DataPipelineBuilder:
        """Read WRD file containing text transcriptions."""
        wrd_file = self._manifest_dir.joinpath(f"{split}.wrd")

        return read_text(wrd_file, key="text", rtrim=True, memory_map=True)

    @property
    def splits(self) -> set[str]:
        """Return the set of available splits."""
        return self._splits

    @staticmethod
    def to_seq2seq_batch(example: dict[str, Any]) -> Seq2SeqBatch:
        """Convert collated example to Seq2SeqBatch."""
        audio_data = example["audio"]["data"]["waveform"]
        text_data = example["text"]

        return Seq2SeqBatch(
            source_seqs=audio_data["seqs"],
            source_seq_lens=audio_data["seq_lens"],
            target_seqs=text_data["seqs"],
            target_seq_lens=text_data["seq_lens"],
            example=example,
        )

    def create_reader(
        self,
        split: str,
        tokenizer: Tokenizer,
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
        seed: int,
        max_num_batches: int | None,
        cached_fd_count: int,
        sync_mode: SyncMode,
        # Provided by TrainerSection
        num_accumulate: int,
    ) -> DataReader[Seq2SeqBatch]:
        """Create data reader with complete audio+text processing pipeline."""
        if split not in self._splits:
            raise DataReadError(
                f"Unknown split: {split}. Available splits: {self._splits}",
            )

        # Read audio directory path from the first line of the manifest
        audio_dir = self._retrieve_audio_directory(split)

        # Read TSV manifest -> { audio: str, audio_size: int }
        tsv_pipeline = self._read_manifest(split).and_return()

        # Read WRD text file
        wrd_pipeline = self._read_wrd_file(split).and_return()

        # Zip TSV and WRD pipelines
        builder = DataPipeline.zip(
            pipelines=[tsv_pipeline, wrd_pipeline], names=["tsv", "wrd"]
        )

        def flatten_example(example: dict[str, Any]) -> dict[str, Any]:
            """Flatten TSV+WRD structure into single example."""
            return {**example["tsv"], **example["wrd"]}  # Merge TSV and WRD data

        # Flattening -> { audio: str, audio_size: int, text: str }
        builder.map(flatten_example)

        # Filter to match audio size expectations
        builder = filter_by_min_max_audio_size(
            builder,
            min_audio_len=min_audio_len,
            max_audio_len=max_audio_len,
            audio_size_selector="audio_size",
        )

        # Shuffle the dataset samples
        if example_shuffle_window != 1:
            builder.shuffle(example_shuffle_window, seed)
            seed += 1

        # Shard across distributed processes
        if gangs.dp.size > 1:
            builder.shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)
            seed += gangs.dp.rank

        # Length batching -> list( { audio: str, audio_size: int, text: str } )
        builder = add_length_batching(
            builder,
            min_audio_len=min_audio_len,
            max_audio_len=max_audio_len,
            max_num_elements=max_num_elements,
            num_seqs_multiple_of=num_seqs_multiple_of,
            drop_remainder=drop_remainder,
            selector="audio_size",
        )

        # Shuffle batches (to get randomized lengths)
        builder = add_batch_shuffling(
            builder, batch_shuffle_window=batch_shuffle_window, seed=seed
        )
        seed += 1

        # Load audios -> list( { audio.path: str, audio.data: MemoryBlock, audio_size: int, text: str } )
        builder = add_audio_file_loading(
            builder,
            audio_dir=audio_dir,
            cached_fd_count=cached_fd_count,
            selector="[*].audio",
        )
        # Audio decoding -> list( { audio.path: str, audio.data.sample_rate: int, audio.data.format: int,
        #                           audio.data.waveform: tensor, audio_size: int, text: str } )
        builder = add_audio_decoding(
            builder,
            dtype=dtype,
            normalize_audio=normalize_audio,
            npc=npc,
            selector="[*].audio.data",
        )
        if normalize_audio:
            builder = add_layernorm(
                builder, dtype=dtype, selector="[*].audio.data.waveform"
            )

        # Text encoding -> list( { audio.path: str, audio.data.sample_rate: int, audio.data.format: int,
        #                          audio.data.waveform: tensor, audio_size: int, text: tensor } )
        token_encoder = tokenizer.create_encoder()
        builder = encode_text(
            builder, token_encoder=token_encoder, npc=npc, selector="[*].text"
        )

        # Collating -> { audio.path: list(str), audio.data.sample_rate: list(int), audio.data.format: list(int),
        #                audio.data.waveform.is_ragged: bool, audio.data.waveform.seqs: [tensor],
        #                audio.data.waveform.seq_lens: list(int), audio_size: int,
        #                text.is_ragged: bool, text.seqs: [tensor], text.seq_lens: list(int) }
        pad_idx = tokenizer.vocab_info.pad_idx
        if pad_idx is None:
            raise ValueError(
                f"Tokenizer must have a pad_idx for ASR training but consists of {tokenizer.vocab_info}."
            )
        builder = collate_with_pad_ix(
            builder, pad_idx=pad_idx, no_padding=False, selector="text"
        )

        # Limit batches
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch
        builder.prefetch(num_prefetch)

        # Wrap samples in Seq2Seq
        builder.map(Wav2Vec2AsrDataset.to_seq2seq_batch)

        pipeline = builder.and_return()

        return DataPipelineReader[Seq2SeqBatch](
            pipeline,
            gangs,
            num_accumulate=num_accumulate,
            sync=True,
            sync_mode=sync_mode,
        )


@dataclass
class Wav2Vec2AsrDatasetConfig:
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


def open_wav2vec2_asr_dataset(config: Wav2Vec2AsrDatasetConfig) -> Wav2Vec2AsrDataset:
    """The mapping between the dataset asset card definition and the Wav2Vec2AsrDataset."""
    return Wav2Vec2AsrDataset.from_path(config.manifest_dir)
