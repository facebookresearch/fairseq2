# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, List, Set, final

import pyarrow as pa
import pyarrow.parquet as pq
from torch import Tensor

from fairseq2.data.data_pipeline import DataPipelineBuilder, read_sequence
from fairseq2.data.parquet import (
    FragmentLoadingConfig,
    FragmentStreamingConfig,
    NamedColumns,
    ParquetFragmentLoader,
    ParquetFragmentStreamer,
)
from fairseq2.data.tokenizers import TokenEncoder
from fairseq2.datasets import DataPipelineReader, DataReader, SequenceBatch, SyncMode
from fairseq2.dependency import DependencyResolver
from fairseq2.gang import Gang
from fairseq2.nn import BatchLayout

from .dataset import TextReadOptions

PARQUET_TEXT_DATASET_FAMILY: Final = "parquet_text"


@dataclass
class DefaultTextSchema(NamedColumns):
    text: str = "text"
    extra_columns: List[str] | None = None


class ParquetDatasetInterface:

    _name: str
    _dataset: pq.ParquetDataset
    _splits: set[str]
    split_column: str = "split"

    def __init__(self, name: str, dataset: pq.ParquetDataset, splits: set[str]) -> None:
        self._dataset = dataset
        self._splits = splits
        self._name = name

    @classmethod
    def from_path(
        cls,
        resolver: DependencyResolver,
        path: Path | str | List[str | Path],
        name: str,
        filesystem: Any | None = None,
    ) -> "ParquetDatasetInterface":

        # from stopes.fb_config import get_filesystem_from_path
        # if filesystem is None:
        #     path, filesystem = get_filesystem_from_path(path)
        dataset = pq.ParquetDataset(path, filesystem=filesystem)  # type: ignore

        assert isinstance(dataset, pq.ParquetDataset)
        partition_columns: List[str] = []
        if dataset.partitioning is not None:
            partition_columns = dataset.partitioning.schema.names

        splits: Set[str] = set()
        if dataset.partitioning is not None and cls.split_column in partition_columns:
            idx = partition_columns.index(cls.split_column)
            _splits = dataset.partitioning.dictionaries[idx]
            if _splits is None:
                splits = set()
            else:
                splits = set(_splits.to_pylist())

        return cls(name, dataset, splits)

    def splits(self) -> set[str]:
        return self._splits


@final
class ParquetTextDataset(ParquetDatasetInterface):

    @staticmethod
    def get_example_loading_builder(
        dataset: pq.ParquetDataset,
        options: TextReadOptions,
        split: str,
        columns: NamedColumns | None,
        seed: int,
        rank: int,
        world_size: int,
        pa_cpu_count: int = 20,
    ) -> DataPipelineBuilder:

        npc = options.npc
        pa_cpu_count = int(options.extras.get("pa_cpu_count", pa_cpu_count))  # type: ignore
        pa.set_cpu_count(pa_cpu_count)
        pa.set_io_thread_count(pa_cpu_count)

        # Streaming
        partition_filters = options.extras.get("partition_filters", None)
        parquet_files: List[str] = dataset.files  # type: ignore

        is_train_streaming = (
            "train" in split and options.sync_mode == SyncMode.UNTIL_FIRST
        )  # FIXME: make it configurable

        files_circular_shift = options.extras.get("files_circular_shift", False)
        assert isinstance(files_circular_shift, bool)

        fragment_shuffle_window = options.extras.get(
            "fragment_shuffle_window", -1 if is_train_streaming else 0
        )
        assert isinstance(fragment_shuffle_window, int)

        fragment_config = FragmentStreamingConfig(
            parquet_path=parquet_files,
            filesystem=dataset.filesystem,
            nb_epochs=(None if is_train_streaming else 1),
            partition_filters=partition_filters,  # type: ignore
            split_to_row_groups=True,
            files_circular_shift=files_circular_shift,
            seed=seed,
            fragment_shuffle_window=fragment_shuffle_window,
        )

        fragment_config = fragment_config.add_partition_filter(
            pa.compute.field("split") == split
        )
        fragement_builder = ParquetFragmentStreamer(
            config=fragment_config
        ).build_pipeline(rank=rank, world_size=world_size)

        num_parallel_fragments = options.extras.get("num_parallel_fragments", npc)
        assert isinstance(num_parallel_fragments, int)
        assert num_parallel_fragments > 0, "num_parallel_fragments must be > 0"

        columns = options.extras.get("columns", columns)  # type: ignore
        assert columns is None or isinstance(columns, NamedColumns)

        cache = options.extras.get("cache", False)
        assert isinstance(cache, bool)

        add_fragment_traces = options.extras.get("add_fragment_traces", False)
        assert isinstance(add_fragment_traces, bool)

        loading_config = FragmentLoadingConfig(
            columns=columns,
            add_fragment_traces=add_fragment_traces,
            num_parallel_fragments=num_parallel_fragments,
            nb_prefetch=options.num_prefetch,
            non_deterministic_read=True,
            cache=cache,
            drop_null=False,
            filters=None,
        )

        # load data in memory
        builder = ParquetFragmentLoader(config=loading_config).apply(fragement_builder)

        builder = builder.yield_from(
            lambda table: read_sequence(table.to_pylist()).and_return()
        )
        return builder

    def create_reader(
        self,
        text_encoder: TokenEncoder,
        pad_idx: int | None,
        gang: Gang,
        # TODO: remove?
        min_seq_len: int,
        max_seq_len: int,
        split: str = "train",  # FIXME
        options: TextReadOptions | None = None,
    ) -> DataReader[SequenceBatch]:
        if options is None:
            options = TextReadOptions()

        npc = options.npc
        seed = options.seed

        builder = ParquetTextDataset.get_example_loading_builder(
            self._dataset,
            options,
            split=split,
            columns=DefaultTextSchema(),
            seed=seed,
            rank=gang.rank,
            world_size=gang.size,
        )

        seed = options.seed + gang.rank
        # this can important if dataset was not shuffled
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed)
            seed += 1

        def encode(example: dict[str, Any]) -> Tensor:
            return text_encoder(example["text"])

        # this requires GIL, so we do it in the main thread
        builder.map(encode, num_parallel_calls=1)

        bsz = 2
        sl = 8192
        #        builder.pack((2 * 8192) + 1, 8192, truncate=True, pinned_memory=True)
        builder.pack((bsz * sl) + 1, sl, truncate=True, pinned_memory=True)

        BatchLayout.compiled_max_seq_len = sl

        seed += 1

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Convert to `SequenceBatch`.
        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            seqs, seq_lens = example["seqs"], example["seq_lens"]

            return SequenceBatch(seqs, seq_lens, packed=True)

        pipeline = builder.map(to_batch).and_return()
        pipeline = builder.prefetch(100).and_return()

        return DataPipelineReader[SequenceBatch](
            self._name, "default", pipeline, gang, options
        )
