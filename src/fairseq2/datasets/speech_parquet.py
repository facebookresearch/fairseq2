# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Set, final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from typing_extensions import override

from fairseq2.data import (
    Collater,
    MemoryBlock,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.data.audio import (
    AudioDecoder,
    AudioDecoderOutput,
    WaveformToFbankConverter,
)
from fairseq2.data.parquet import (
    FragmentLoadingConfig,
    FragmentStreamingConfig,
    NamedColumns,
    ParquetFragmentLoader,
    ParquetFragmentStreamer,
)
from fairseq2.datasets import (
    DataPipelineReader,
    LengthBatching,
    StaticBatching,
    UnknownSplitError,
)
from fairseq2.datasets.speech import (
    AudioCropper,
    SpeechDataset,
    SpeechReadOptions,
    postprocess,
    to_batch,
)
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang
from fairseq2.logging import log
from fairseq2.models.sequence import SequenceBatch


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

    nb_samples_per_fragment = 1000  # FIXME: detect it from the dataset metadata
    max_num_batches: int = 1000
    max_num_examples: int = 50_000
    pa_cpu_count: int = 20

    def __init__(self, name: str, dataset: pq.ParquetDataset, splits: set[str]) -> None:
        self._dataset = dataset
        self._splits = splits
        self._name = name

    @classmethod
    def from_path(
        cls, path: Path | str | List[str | Path], name: str
    ) -> GenericSpeechParquetDataset:
        # to work with BlobStore filesystem
        from stopes.fb_config import get_filesystem_from_path  # type: ignore

        fixed_path, filesystem = get_filesystem_from_path(path)
        datasest = pq.ParquetDataset(fixed_path, filesystem=filesystem)  # type: ignore

        assert isinstance(datasest, pq.ParquetDataset)
        partition_columns: List[str] = []
        if datasest.partitioning is not None:
            partition_columns = datasest.partitioning.schema.names

        splits: Set[str] = set()
        if datasest.partitioning is not None and cls.split_column in partition_columns:
            idx = partition_columns.index(cls.split_column)
            _splits = datasest.partitioning.dictionaries[idx]
            if _splits is None:
                splits = set()
            else:
                splits = set(_splits.to_pylist())

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
        assert min_audio_len <= max_audio_len, "min_audio_len must be <= max_audio_len"

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

        pa_cpu_count = int(options.extras.get("pa_cpu_count", self.pa_cpu_count))  # type: ignore
        pa.set_cpu_count(pa_cpu_count)
        pa.set_io_thread_count(pa_cpu_count)

        # Streaming

        partition_filters = options.extras.get("partition_filters", None)
        parquet_files: List[str] = self._dataset.files  # type: ignore
        fragment_config = FragmentStreamingConfig(
            parquet_path=parquet_files,
            filesystem=self._dataset.filesystem,
            nb_epochs=(None if split == "train" else 1),
            partition_filters=partition_filters,  # type: ignore
            split_to_row_groups=True,
            files_circular_shift=True,
            seed=seed,
            fragment_shuffle_window=max(
                100, options.example_shuffle_window // self.nb_samples_per_fragment
            ),
        )
        fragement_builder = ParquetFragmentStreamer(
            config=fragment_config
        ).build_pipeline(rank=gang.rank, world_size=gang.size)

        seed += gang.rank

        loading_config = FragmentLoadingConfig(
            columns=DefaultAudioSchema(),
            add_fragment_traces=False,
            num_parallel_fragments=npc,
            nb_prefetch=options.num_prefetch,
            # non_deterministic_read=True,
            drop_null=False,
        )

        # load data in memory
        builder = ParquetFragmentLoader(config=loading_config).apply(fragement_builder)

        # dispatch table into examples
        builder = builder.yield_from(
            lambda table: read_sequence(
                table.to_pandas().to_dict(orient="records")
            ).and_return()
        )

        # truncate length to max_audio_len
        builder = builder.filter(lambda x: int(x["length"]) < min_audio_len)
        builder = builder.map(lambda x: min(max_audio_len, x), selector="length")

        # shuffle examples in memory
        if options.example_shuffle_window != 1:
            # FIXME: make it configurable, we need some upper bound to avoid OOM in cpu
            example_shuffle_window = min(
                options.example_shuffle_window, self.max_num_examples
            )
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
            batch_shuffle_window = min(
                options.batch_shuffle_window, self.max_num_batches
            )
            builder.shuffle(batch_shuffle_window, seed)
            seed += 1

        # Decode audio.
        audio_decoder = AudioDecoder(
            dtype=torch.float32 if options.normalize_audio else options.dtype
        )

        def decoded_audio(_bytes: NDArray[np.int8]) -> AudioDecoderOutput:
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

            builder.map(fbank_converter, selector="[*].audio", num_parallel_calls=npc)
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
        collater = Collater(pad_value=None if no_padding else 0)
        builder.map(collater)

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        builder.prefetch(options.num_prefetch)

        pipeline = builder.map(
            partial(to_batch, no_padding=no_padding, device=gang.device)
        ).and_return()

        return DataPipelineReader[SequenceBatch](
            self._name, split, pipeline, gang, options, strict_state=False
        )
