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
    DataPipelineBuilder,
    MemoryBlock,
    read_sequence,
)
from fairseq2.data.audio import (
    AudioDecoder,
    AudioDecoderOutput,
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
    UnknownSplitError,
)
from fairseq2.datasets.speech import (
    GenericSpeechDataset,
    SpeechDataset,
    SpeechReadOptions,
)
from fairseq2.gang import Gang
from fairseq2.logging import log
from fairseq2.models.sequence import SequenceBatch

PARQUET_SPEECH_DATASET_FAMILY: Final = "generic_parquet_speech"


@dataclass
class DefaultAudioSchema(NamedColumns):
    audio: str = "audio_bytes"
    length: str = "length"
    size: str | None = "size"  # size of the audio in bytes : len(audio_bytes)
    extra_columns: List[str] | None = None


class ParquetDatasetInterface:

    _name: str
    _dataset: pq.ParquetDataset
    _splits: set[str]
    split_column: str = "split"

    max_num_batches: int = 1000
    max_num_examples: int = 2_000_000

    def __init__(self, name: str, dataset: pq.ParquetDataset, splits: set[str]) -> None:
        self._dataset = dataset
        self._splits = splits
        self._name = name

    @classmethod
    def from_path(
        cls,
        path: Path | str | List[str | Path],
        name: str,
        filesystem: Any | None = None,
    ) -> "ParquetDatasetInterface":

        # from stopes.fb_config import get_filesystem_from_path
        # path, filesystem = get_filesystem_from_path(path)  # type: ignore
        datasest = pq.ParquetDataset(path, filesystem=filesystem)  # type: ignore

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

        return cls(name, datasest, splits)

    def splits(self) -> set[str]:
        return self._splits


@final
class GenericSpeechParquetDataset(ParquetDatasetInterface, SpeechDataset):
    """Represents a generic parquet-based Speech dataset."""

    @staticmethod
    def rename_feature(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for example in batch:
            if "fbank" in example["audio"]:
                example["audio_feature"] = example["audio"].pop("fbank")
            elif "waveform" in example["audio"]:
                example["audio_feature"] = example["audio"].pop("waveform")
        return batch

    @staticmethod
    def add_audio_decoding(
        builder: DataPipelineBuilder, options: SpeechReadOptions
    ) -> DataPipelineBuilder:

        audio_decoder = AudioDecoder(
            dtype=torch.float32 if options.normalize_audio else options.dtype
        )

        def decoded_audio(_bytes: NDArray[np.int8]) -> AudioDecoderOutput:
            return audio_decoder(MemoryBlock(_bytes.tobytes()))

        builder.map(decoded_audio, selector="[*].audio", num_parallel_calls=options.npc)

        return builder

    @staticmethod
    def get_example_loading_builder(
        dataset: pq.ParquetDataset,
        options: SpeechReadOptions,
        split: str,
        columns: NamedColumns | None,
        seed: int,
        rank: int,
        world_size: int,
        pa_cpu_count: int = 20,
        nb_samples_per_fragment: int = 1000,
    ) -> DataPipelineBuilder:

        npc = options.npc
        pa_cpu_count = int(options.extras.get("pa_cpu_count", pa_cpu_count))  # type: ignore
        pa.set_cpu_count(pa_cpu_count)
        pa.set_io_thread_count(pa_cpu_count)

        # Streaming
        partition_filters = options.extras.get("partition_filters", None)
        parquet_files: List[str] = dataset.files  # type: ignore
        fragment_config = FragmentStreamingConfig(
            parquet_path=parquet_files,
            filesystem=dataset.filesystem,
            nb_epochs=(None if split == "train" else 1),
            partition_filters=partition_filters,  # type: ignore
            split_to_row_groups=True,
            files_circular_shift=True,
            seed=seed,
            fragment_shuffle_window=max(
                100, options.example_shuffle_window // nb_samples_per_fragment
            ),
        )
        fragement_builder = ParquetFragmentStreamer(
            config=fragment_config
        ).build_pipeline(rank=rank, world_size=world_size)

        # we want to give less compute to the loader compared to the audio decoder
        num_parallel_fragments = options.extras.get(
            "num_parallel_fragments", max(npc // 3, 1)
        )
        assert isinstance(num_parallel_fragments, int)
        assert num_parallel_fragments > 0, "num_parallel_fragments must be > 0"

        columns = options.extras.get("columns", columns)  # type: ignore
        assert columns is None or isinstance(columns, NamedColumns)

        loading_config = FragmentLoadingConfig(
            columns=columns,
            add_fragment_traces=False,
            num_parallel_fragments=num_parallel_fragments,
            nb_prefetch=options.num_prefetch,
            non_deterministic_read=True,
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
        no_padding = options.no_padding

        options.batch_shuffle_window = min(
            options.batch_shuffle_window, self.max_num_batches
        )
        builder = self.get_example_loading_builder(
            self._dataset,
            options,
            split,
            columns=DefaultAudioSchema(),
            seed=seed,
            rank=gang.rank,
            world_size=gang.size,
        )
        seed += gang.size
        # truncate length to max_audio_len
        builder = builder.filter(lambda x: int(x["length"]) >= min_audio_len)
        builder = builder.map(lambda x: min(max_audio_len, x), selector="length")

        # shuffle examples in memory
        if options.example_shuffle_window != 1:
            # FIXME: make it configurable, we need some upper bound to avoid OOM in cpu
            example_shuffle_window = min(
                options.example_shuffle_window, self.max_num_examples
            )
            builder = builder.prefetch(int(2 * example_shuffle_window))
            builder = builder.shuffle(example_shuffle_window, seed=seed)
            seed += 1

        builder = GenericSpeechDataset.add_bucketing_pipeline(
            builder,
            options,
            max_audio_len=max_audio_len,
            min_audio_len=min_audio_len,
            seed=seed,
            columns="length",
        )
        seed += 1

        builder = GenericSpeechParquetDataset.add_audio_decoding(builder, options)
        builder = GenericSpeechDataset.audio_post_process(
            builder, options, GenericSpeechParquetDataset.rename_feature
        )
        builder = GenericSpeechDataset.add_audio_cropping(
            builder, options, seed, max_audio_len
        )

        # Collate batched examples into a batch.
        collater = Collater(pad_value=None if no_padding else 0)
        builder.map(collater)

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        builder.prefetch(options.num_prefetch)

        pipeline = builder.map(
            partial(
                GenericSpeechDataset.to_batch, no_padding=no_padding, device=gang.device
            )
        ).and_return()

        return DataPipelineReader[SequenceBatch](
            self._name, split, pipeline, gang, options, strict_state=False
        )
