# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, final

import torch

from fairseq2.data.data_pipeline import (
    DataPipeline,
    DataPipelineBuilder,
)
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DataReadError,
    DatasetOpenError,
    Seq2SeqBatch,
    SyncMode,
)
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.recipe.config import DatasetSection

from .batch_utils import (
    BatchingPipeline,
    BatchingStrategy,
)
from .preprocessing import AudioProcessingPipeline, TextProcessingPipeline

WAV2VEC2_ASR_DATASET: Final = "wav2vec2_asr"


@dataclass(kw_only=True)
class Wav2Vec2AsrDatasetSection(DatasetSection):
    """wav2vec2 ASR dataset configuration section."""

    name: str | None = "librilight_asr_10h"
    """The name, path or path to the asset card of the speech dataset."""

    family: str = WAV2VEC2_ASR_DATASET  # type: ignore

    path: Path | None = None
    """The path of the directory with a `.tsv` manifest and `.wrd` text transcriptions. Populated through the asset system via `open_wav2vec2_asr_dataset`."""

    train_split: str | None = "train"
    """The name of the training data split. Expecting a {train_split}.tsvfile in the dataset directory. Only `None` during evaluation.
    """

    valid_split: str | None = "dev_other"
    """The name of the validation data split(s). Expecting a {valid_split}.tsv and {valid_split}.wrd files in the dataset directory. Format multiple splits interspersed by `,`and without spaces (`'valid,dev_clean,test_clean'`).
    """

    min_audio_len: int = 1
    """The minimum audio sequence length."""

    max_audio_len: int = 800_000
    """The maximum audio sequence length."""

    # Batching parameters
    batching_strategy: BatchingStrategy = BatchingStrategy.LENGTH
    """Batching strategy is defined through an enum:
    - BatchingStrategy.LENGTH ("LENGTH") = Specifies batching where each batch has a maximum number of elements.
    - BatchingStrategy.STATIC ("STATIC") = Specifies batching where each batch has the same number of examples.
    """

    batch_size: int | None = None
    """If `batching_strategy = BatchingStrategy.STATIC`, ignores `max_num_tokens` and each batch will have `batch_size` examples.
    """

    num_seqs_multiple_of: int = 8
    """If `batching_strategy = BatchingStrategy.LENGTH, ignores `batch_size` and each batch will have
    `<= max_num_elements` elements with `<= num_seqs_multipe_of` or `num_seqs_multiple_of * N` sequences.
    This is primarily for hardware optimization, but will only work when sufficient enough sequences are available for bucketing.
    """

    max_num_elements: int = 3_200_000
    """If `batching_strategy = BatchingStrategy.LENGTH, ignores `batch_size` and each batch will have
    `<= max_num_elements` elements with `<= num_seqs_multipe_of` or `num_seqs_multiple_of * N` sequences.
    This is primarily for hardware optimization, but will only work when sufficient enough sequences are available for bucketing.
    """

    # Audio processing
    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    no_padding: bool = True
    """If ``True``, all elements in the batch will be truncated to by batch minimal length.
    Therefore, no padding will be applied to the batch.
    """

    npc: int = 10
    """The number of parallel calls to use in the pipeline."""

    # Shuffling parameters
    example_shuffle_window: int = 0
    """The size of the sliding window for shuffling examples (pre-batch)."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

    # Batching behavior
    drop_remainder: bool = False
    """If ``True``, drops the last set of batches if they have in total fewer examples than requested."""

    sync_batches: bool = True
    """If ``True``, ensures that each process reads the same number of batches."""

    sync_mode: SyncMode = SyncMode.UNTIL_FIRST
    """The data synchronization mode among processes."""

    # Misc
    max_num_batches: int | None = None
    """The maximum number of batches to return."""

    num_accumulate: int = 4
    """The number of batches to accumulate in each iteration. This option needs to be
    synchronized with trainer.grad_accumulation.num_batches (default=1) to work."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    seed: int = 2
    """The seed to initialize the random number generators used internally."""

    cached_fd_count: int = 1000
    """Enables an LRU cache on the last ``cached_fd_count`` files read.
    ``FileMapper`` will memory map all the cached file, so this is especially
    useful for reading several slices of the same file.
    """


@final
class Wav2Vec2AsrDataset:
    """wav2vec2 ASR dataset with audio decoding and text tokenization support."""

    _name: str
    _manifest_dir: Path
    _splits: set[str]

    def __init__(self, name: str, manifest_dir: Path, splits: set[str]) -> None:
        self._name = name
        self._manifest_dir = manifest_dir
        self._splits = splits

    @classmethod
    def from_path(cls, path: Path, name: str) -> "Wav2Vec2AsrDataset":
        """Create dataset from path (file or directory)."""
        path = path.expanduser().resolve()

        if not path.is_dir():
            return cls(name, manifest_dir=path.parent, splits={path.stem})

        try:
            splits = {f.stem for f in path.glob("*.tsv")}
        except OSError as ex:
            raise DatasetOpenError(
                name,
                f"The splits under the '{path}' directory of the '{name}' dataset cannot be determined. See the nested exception for details.",
            ) from ex

        if not splits:
            raise DatasetOpenError(
                name,
                f"The '{path}' directory of the '{name}' dataset does not contain any splits.",
            )

        return cls(name, manifest_dir=path, splits=splits)

    def _retrieve_audio_directory(self, split: str) -> Path | None:
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
                self._name,
                split,
                f"The {manifest_file} manifest file cannot be read. See the nested exception for details.",
            ) from ex

        try:
            audio_dir = Path(header)
            if audio_dir.exists():
                return audio_dir
            else:
                raise ValueError  # TODO: (cirquit) Test this
        except ValueError:
            raise DataReadError(
                self._name,
                split,
                f"The first line of the '{manifest_file}' manifest file must point to a data directory.",
            ) from None

    def _read_manifest(
        self, split: str, audio_dir: Path | None, min_audio_len: int, max_audio_len: int
    ) -> DataPipelineBuilder:
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
            block_size=10 * 1024 * 1024,
        )

        if audio_dir is not None:
            builder.skip(1)  # Path to the data directory

        field_splitter = StrSplitter(names=["audio", "audio_size"])
        builder.map(field_splitter)

        # Convert audio_size to int and clamp to max_audio_len
        builder.map(
            lambda x: min(int(x), max_audio_len),
            selector="audio_size",
        )

        # Filter by minimum length
        return builder.filter(lambda sample: sample["audio_size"] >= min_audio_len)

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
        batching_strategy: BatchingStrategy,
        batch_size: int | None,
        max_num_elements: int,
        num_seqs_multiple_of: int,
        # Audio processing
        dtype: torch.dtype,
        normalize_audio: bool,
        no_padding: bool,
        npc: int,
        # Shuffling
        example_shuffle_window: int,
        batch_shuffle_window: int,
        # Misc
        num_accumulate: int,
        num_prefetch: int,
        drop_remainder: bool,
        sync_batches: bool,
        sync_mode: SyncMode,
        seed: int,
        max_num_batches: int | None,
        cached_fd_count: int,
    ) -> DataReader[Seq2SeqBatch]:
        """Create data reader with complete audio+text processing pipeline."""
        if split not in self._splits:
            raise DataReadError(
                self._name,
                split,
                f"Unknown split: {split}. Available splits: {self._splits}",
            )

        # Read audio directory path from the first line of the manifest
        audio_dir = self._retrieve_audio_directory(split)

        # Read TSV manifest
        tsv_pipeline = self._read_manifest(
            split, audio_dir, min_audio_len, max_audio_len
        ).and_return()

        # Read WRD text file
        wrd_pipeline = self._read_wrd_file(split).and_return()

        # Zip TSV and WRD pipelines
        builder = DataPipeline.zip(
            pipelines=[tsv_pipeline, wrd_pipeline], names=["tsv", "wrd"]
        )

        def flatten_example(example: Dict[str, Any]) -> Dict[str, Any]:
            """Flatten TSV+WRD structure into single example."""
            return {**example["tsv"], **example["wrd"]}  # Merge TSV and WRD data

        # flattening -> { audio: str, audio_size: int, text: str }
        builder.map(flatten_example)

        # Shuffle the dataset samples
        if example_shuffle_window != 1:
            builder.shuffle(example_shuffle_window, seed)
            seed += 1

        # Shard across distributed processes
        if gangs.dp.size > 1:
            builder.shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)
            seed += gangs.dp.rank

        # Batching pipeline - works on metadata only
        batch_pipeline = BatchingPipeline()
        if batching_strategy == BatchingStrategy.STATIC:
            # Note that we filter by audio_length here while the w2v2 train code
            # applies the functionally similar `add_audio_cropping` but **does not** filter by min_audio_length
            # This interplays with the required padding at the collator to stack samples
            builder = batch_pipeline.filter_by_min_max_audio_length(
                builder, min_audio_len, max_audio_len
            )
            builder = batch_pipeline.add_static_batching(
                builder, batch_size, drop_remainder  # type: ignore
            )
        else:
            # Length batching -> list( { audio: str, audio_size: int, text: str } )
            builder = batch_pipeline.add_length_batching(
                builder,
                min_audio_len,
                max_audio_len,
                max_num_elements,
                num_seqs_multiple_of,
                drop_remainder,
            )

        # Shuffle batches (to get randomized lengths)
        builder = batch_pipeline.add_batch_shuffling(
            builder, batch_shuffle_window, seed
        )
        seed += 1

        audio_pipeline = AudioProcessingPipeline()
        # Path resolution -> list( { audio.path: str, audio.data: MemoryBlock, audio_size: int, text: str } )
        builder = audio_pipeline.add_path_resolution(
            builder, audio_dir, cached_fd_count
        )
        # Audio decoding -> list( { audio.path: str, audio.data.sample_rate: int, audio.data.format: int,
        #                           audio.data.waveform: tensor, audio_size: int, text: str } )
        builder = audio_pipeline.add_audio_decoding(
            builder, dtype, normalize_audio, npc
        )
        if normalize_audio:
            builder = audio_pipeline.add_layernorm(builder, dtype)

        text_pipeline = TextProcessingPipeline(tokenizer)
        # Text encoding -> list( { audio.path: str, audio.data.sample_rate: int, audio.data.format: int,
        #                          audio.data.waveform: tensor, audio_size: int, text: tensor } )
        builder = text_pipeline.encode_text(builder, npc)

        if no_padding:
            log.warning(
                "Collating without padding is currently not supported, defaulting to padding."
            )

        # Collating -> { audio.path: list(str), audio.data.sample_rate: list(int), audio.data.format: list(int),
        #                audio.data.waveform.is_ragged: bool, audio.data.waveform.seqs: [tensor],
        #                audio.data.waveform.seq_lens: list(int), audio_size: int,
        #                text.is_ragged: bool, text.seqs: [tensor], text.seq_lens: list(int) }
        builder = text_pipeline.collate_with_pad_ix(
            builder, no_padding=False
        )  # no_padding)

        # Limit batches
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch
        builder.prefetch(num_prefetch)

        # Wrap samples in Seq2Seq
        builder.map(Wav2Vec2AsrDataset.to_seq2seq_batch)

        pipeline = builder.and_return()

        return DataPipelineReader[Seq2SeqBatch](
            self._name,
            split,
            pipeline,
            gangs,
            num_accumulate=num_accumulate,
            sync=sync_batches,
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
      data: (all keys here must have a companion parameter in this config)
    ```
    """

    data: Path = field(default_factory=Path)


def open_wav2vec2_asr_dataset(
    name: str, config: Wav2Vec2AsrDatasetConfig
) -> Wav2Vec2AsrDataset:
    """The mapping between the dataset asset card definition and the Wav2Vec2AsrDataset."""
    return Wav2Vec2AsrDataset.from_path(config.data, name)
