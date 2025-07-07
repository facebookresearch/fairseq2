# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, List, Set, final

import pyarrow as pa
import pyarrow.parquet as pq
from typing_extensions import override

from fairseq2.data import DataPipelineBuilder, read_sequence
from fairseq2.data.parquet import (
    FragmentLoadingConfig,
    FragmentStreamingConfig,
    NamedColumns,
    ParquetFragmentLoader,
    ParquetFragmentStreamer,
)
from fairseq2.data.text.tokenizers import TextTokenEncoder
from fairseq2.datasets import DataPipelineReader, DataReader, SequenceBatch, SyncMode
from fairseq2.datasets.jsonl import JsonlDataset
from fairseq2.datasets.text import TextDataset, TextReadOptions
from fairseq2.gang import Gang
from fairseq2.logging import log

PARQUET_TEXT_DATASET_FAMILY: Final = "parquet_text"


@dataclass
class DefaultTextSchema(NamedColumns):
    """Default schema for parquet text datasets.
    One should pass an different `text` value here if the parquet file has a different column name.

    Or, alternatively, one can pass it with
    options.extras["columns"] = DefaultTextSchema(text="my_text_column_name")
    """

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
class ParquetTextDataset(ParquetDatasetInterface, TextDataset):

    @staticmethod
    def get_example_loading_builder(
        dataset: pq.ParquetDataset,
        options: TextReadOptions,
        split: str | None,
        columns: NamedColumns | None,
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

        is_train_streaming = (split is not None) and (
            "train" in split and options.sync_mode == SyncMode.UNTIL_FIRST
        )  # FIXME: make it configurable

        files_circular_shift = options.extras.get("files_circular_shift", False)
        assert isinstance(
            files_circular_shift, bool
        ), "files_circular_shift must be bool"

        fragment_shuffle_window = options.extras.get(
            "fragment_shuffle_window", -1 if is_train_streaming else 0
        )
        assert isinstance(
            fragment_shuffle_window, int
        ), "fragment_shuffle_window must be int"

        fragment_config = FragmentStreamingConfig(
            parquet_path=parquet_files,
            filesystem=dataset.filesystem,
            nb_epochs=(None if is_train_streaming else 1),
            partition_filters=partition_filters,  # type: ignore
            split_to_row_groups=True,
            files_circular_shift=files_circular_shift,
            seed=options.seed,
            fragment_shuffle_window=fragment_shuffle_window,
        )

        if split is not None:
            fragment_config = fragment_config.add_partition_filter(
                pa.compute.field("split") == split
            )
        fragement_builder = ParquetFragmentStreamer(
            config=fragment_config
        ).build_pipeline(rank=rank, world_size=world_size)

        num_parallel_fragments = options.extras.get("num_parallel_fragments", npc)
        assert isinstance(
            num_parallel_fragments, int
        ), "num_parallel_fragments must be int"
        assert num_parallel_fragments > 0, "num_parallel_fragments must be > 0"

        columns = options.extras.get("columns", columns)  # type: ignore
        assert columns is None or isinstance(
            columns, NamedColumns
        ), "columns must be NamedColumns"

        cache = options.extras.get("cache", False)
        assert isinstance(cache, bool), "cache must be bool"

        add_fragment_traces = options.extras.get("add_fragment_traces", False)
        assert isinstance(add_fragment_traces, bool), "add_fragment_traces must be bool"

        loading_config = FragmentLoadingConfig(
            columns=columns,
            add_fragment_traces=add_fragment_traces,
            num_parallel_fragments=num_parallel_fragments,
            nb_prefetch=min(options.num_prefetch, 3 * num_parallel_fragments),
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

    @override
    def create_reader(
        self,
        text_encoder: TextTokenEncoder,
        pad_idx: int | None,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: TextReadOptions | None = None,
        split: str | None = None,
    ) -> DataReader[SequenceBatch]:
        if options is None:
            options = TextReadOptions()
        else:
            options = deepcopy(options)

        if min_seq_len > 0:
            log.warning(
                f"The `min_seq_len={min_seq_len}` is ignored for ParquetTextDataset because of packing."
            )

        builder = ParquetTextDataset.get_example_loading_builder(
            self._dataset,
            options,
            split=split,
            columns=DefaultTextSchema(),
            rank=gang.rank,
            world_size=gang.size,
        )
        options.seed += gang.rank

        pipeline = JsonlDataset.build_pipeline_backend(
            builder,
            options,
            text_encoder,
            pad_idx=pad_idx,
            max_seq_len=max_seq_len,
            text_column_name="text",
            device=gang.device,
        )
        return DataPipelineReader[SequenceBatch](
            self._name, "default", pipeline, gang, options, strict_state=False
        )
