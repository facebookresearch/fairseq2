# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Any, Dict, Final, List, final
import pyarrow.parquet as pq
import pyarrow as pa

from fairseq2.datasets.speech import SpeechDataset, SpeechReadOptions, postprocess, AudioCropper
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
    read_sequence
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



def rename_feature(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for example in batch:
        if "fbank" in example["audio"]:
            example["audio_feature"] = example["audio"].pop("fbank")
        elif "waveform" in example["audio"]:
            example["audio_feature"] = example["audio"].pop("waveform")
    return batch


PARQUET_SPEECH_DATASET_FAMILY: Final = "generic_parquet_speech"


@dataclass
class DefaultAudioSchema(NamedColumns):
    audio: str = "audio_bytes"
    length: str = "length"
    size: str | None = "size"  # size of the audio in bytes : len(audio_bytes)
    extra_columns: List[str] | None = None



@final
class GenericSpeechParquetDataset(SpeechDataset):
    """Represents a generic manifest-based Speech dataset."""

    _name: str
    _dataset: pq.ParquetDataset
    _splits: set[str]
    split_column: str = "split"

    def __init__(self, name: str, dataset: pq.ParquetDataset, splits: set[str]) -> None:
        self._dataset = dataset
        self._splits = splits
        self._name = name

        pa.set_cpu_count(20)
        pa.set_io_thread_count(20)

    @classmethod
    def from_path(cls, path: Path | str | List[str | Path], name: str) -> GenericSpeechParquetDataset:
        # to work with BlobStore filesystem
        from stopes.fb_config import get_filesystem_from_path
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

        return GenericSpeechParquetDataset(name, datasest, splits)

    @override
    def splits(self) -> set[str]:
        return self._splits

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

        # Streaming
        nb_epochs = None if split == "train" else 1
        partition_filters = options.extras.get("partition_filters", None)

        # FIXME: detect it from the dataset
        nb_samples_per_fragment = 1000

        fragment_config = FragmentStreamingConfig(
            parquet_path=self._dataset.files,
            filesystem=self._dataset.filesystem,
            nb_epochs=nb_epochs,
            partition_filters=partition_filters,
            split_to_row_groups=True,
            files_circular_shift=True,
            seed=seed,
            fragment_shuffle_window=max(100, options.example_shuffle_window // nb_samples_per_fragment),
        )
        fragement_builder = ParquetFragmentStreamer(config=fragment_config).build_pipeline(
            rank=gang.rank, world_size=gang.size
        )

        seed += gang.rank

        loading_config = FragmentLoadingConfig(
            columns=DefaultAudioSchema(),
            add_fragment_traces=False,
            num_parallel_fragments=npc,
            nb_prefetch=options.num_prefetch,
            drop_null=False,
        )

        # load data in memory
        builder = ParquetFragmentLoader(config=loading_config).apply(fragement_builder)

        # dispatch table into exampels
        builder = builder.yield_from(
            lambda table: read_sequence(
                table.to_pandas().to_dict(orient="records")
            ).and_return()
        )

        # truncate length to max_audio_len
        builder = builder.map(lambda x: min(max_audio_len, int(x)), selector="length")


        # shuffle examples in memory
        if options.example_shuffle_window != 1:
            example_shuffle_window = min(options.example_shuffle_window, 10_000)
            builder = builder.prefetch(int(1.2 * example_shuffle_window))
            builder = builder.shuffle(example_shuffle_window, seed=seed)
            seed += 1


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
                selector="length",
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
            batch_shuffle_window = min(options.batch_shuffle_window, 100)
            builder.shuffle(batch_shuffle_window, seed)
            seed += 1


        # Decode audio.
        audio_decoder = AudioDecoder(
            dtype=torch.float32 if options.normalize_audio else options.dtype
        )
        def decoded_audio(_bytes):
            return audio_decoder(MemoryBlock(_bytes.tobytes()))

        builder.map(decoded_audio, selector="[*].audio", num_parallel_calls=npc)

        if options.use_fbank:
            fbank_converter = WaveformToFbankConverter(
                num_mel_bins=80,
                waveform_scale=2**15,
                channel_last=True,
                standardize=True,
                dtype=options.dtype,
            )

            builder.map(
                fbank_converter, selector="[*].audio", num_parallel_calls=npc
            )
        else:
            builder.map(
                partial(
                    postprocess,
                    normalize_audio=options.normalize_audio,
                    dtype=options.dtype,
                ),
                selector="[*].audio.waveform",
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
            self._name, split, pipeline, gang, options, strict_state=False
        )
