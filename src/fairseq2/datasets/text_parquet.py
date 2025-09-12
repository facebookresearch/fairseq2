# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, List, Set

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.dataset import get_partition_keys
from tqdm.auto import tqdm
from typing_extensions import override

from fairseq2.data import DataPipeline, DataPipelineBuilder, read_sequence
from fairseq2.data.parquet import (
    FragmentLoadingConfig,
    FragmentStreamingConfig,
    NamedColumns,
    ParquetFragmentLoader,
    ParquetFragmentStreamer,
)
from fairseq2.data.parquet.fragment_streaming.primitives import (
    RejectionDistributionSmoother,
    process_filter,
)
from fairseq2.data.text.tokenizers import TextTokenEncoder
from fairseq2.datasets import DataPipelineReader, DataReader, SequenceBatch
from fairseq2.datasets.jsonl import JsonlDataset
from fairseq2.datasets.text import TextDataset, TextReadOptions
from fairseq2.gang import Gang
from fairseq2.logging import log

PARQUET_TEXT_DATASET_FAMILY: Final = "parquet_text"
WEIGHTED_MIXTURE_PARQUET_DATASET_FAMILY: Final = "weighted_mixture_parquet"


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

    parquet_path_name: str = "_parquet_path"

    def __init__(self, name: str, dataset: pq.ParquetDataset, splits: set[str]) -> None:
        self._dataset = dataset
        self._splits = splits
        self._name = name

    @staticmethod
    def _get_filtered_paths(
        full_partition_df: pa.Table,
        filter_exp: pa.compute.Expression | None,
    ) -> pa.Table:
        """Get dict mapping language partitions to lists of parquet paths after filtering."""
        df = full_partition_df
        if filter_exp is not None:
            df = df.filter(filter_exp)
        return df

    def _get_all_mixture_partitions(self) -> pa.Table:
        """Get table mapping each partition to its parquet path."""
        dicts = []
        for fragment in self._dataset._dataset.get_fragments(
            filter=self._dataset._filter_expression
        ):
            dd = get_partition_keys(fragment.partition_expression) or {}
            dd[self.parquet_path_name] = fragment.path
            dicts.append(dd)
        return pa.Table.from_pylist(dicts)

    @classmethod
    def from_path(
        cls,
        path: Path | str | List[str | Path],
        name: str,
        filesystem: Any | None = None,
    ) -> "ParquetDatasetInterface":

        from stopes.fb_config import get_filesystem_from_path

        path = str(path)
        if ":/" in path and "://" not in path:
            assert path.count(":/") == 1
            path = path.replace(":/", "://")
        log.info(f"Loading parquet dataset from path: {path}")

        if filesystem is None:
            path, filesystem = get_filesystem_from_path(str(path))
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

    @property
    def splits(self) -> Set[str]:
        return self._splits


class ParquetTextDataset(TextDataset, ParquetDatasetInterface):
    """A `TextDataset` configured by `text_parquet.yaml`."""

    @staticmethod
    def _prepare_full_partition_filters(
        partition_filters: Any | None,
        split: str | None,
    ) -> pa.compute.Expression | None:
        """Prepare full partition filters including split filter."""
        full_filters = []

        # Add existing partition filters
        if partition_filters is not None:
            if isinstance(partition_filters, list):
                full_filters.extend(partition_filters)
            else:
                full_filters.append(partition_filters)

        # Add split filter
        if split is not None:
            split_filter = pa.compute.field("split") == split
            full_filters.append(split_filter)

        return process_filter(full_filters if full_filters else None)

    @staticmethod
    def get_example_loading_builder(
        parquet_files: List[str],
        filesystem: Any,
        options: TextReadOptions,
        partition_filters: Any | None,
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
        is_train_streaming = True

        files_circular_shift = options.extras.get("files_circular_shift", True)
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
            filesystem=filesystem,
            nb_epochs=(None if is_train_streaming else 1),
            partition_filters=partition_filters,  # type: ignore
            split_to_row_groups=True,
            files_circular_shift=files_circular_shift,
            seed=options.seed,
            fragment_shuffle_window=fragment_shuffle_window,
        )

        fragement_builder = ParquetFragmentStreamer(
            config=fragment_config
        ).build_pipeline(rank=rank, world_size=world_size)

        smoother_partition_groups = options.extras.get("smoother_partition_groups", [])
        smoother_alpha = options.extras.get("smoother_alpha", 1.0)
        assert isinstance(smoother_partition_groups, list)
        assert isinstance(smoother_alpha, float)

        if smoother_partition_groups and smoother_alpha < 1:
            log.info(
                f"Partition smoothing with alpha={smoother_alpha} and partition_groups={smoother_partition_groups}"
            )
            partition_smoother = RejectionDistributionSmoother(
                partition_groups=smoother_partition_groups,
                alpha=smoother_alpha,  # Partial smoothing between original and uniform
                min_count=100,
                seed=42,
            )
            fragement_builder = fragement_builder.map(partition_smoother).filter(
                lambda x: x is not None
            )

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

        # Apply lowercase transformation to the text field if it exists
        def lowercase_text(example: Dict[str, Any]) -> Dict[str, Any]:
            example["text"] = example["text"].lower()
            return example

        builder = builder.map(lowercase_text)

        return builder

    @staticmethod
    def _build_pipeline_pre_batch_shuffle(
        text_encoder: TextTokenEncoder,
        pad_idx: int | None,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: TextReadOptions,
        parquet_files: List[str],
        filesystem: Any,
        full_partition_filters: Any | None,
        columns: NamedColumns | None = None,
    ) -> DataPipelineBuilder:
        """Build pipeline up to and including pre-batch shuffle processing."""
        if min_seq_len > 0:
            log.warning(
                f"The `min_seq_len={min_seq_len}` is ignored for ParquetTextDataset because of packing."
            )

        options.seed += gang.rank

        builder = ParquetTextDataset.get_example_loading_builder(
            parquet_files=parquet_files,
            filesystem=filesystem,
            options=options,
            partition_filters=full_partition_filters,
            columns=columns or DefaultTextSchema(),
            rank=gang.rank,
            world_size=gang.size,
        )

        builder = JsonlDataset.build_pipeline_pre_batch_shuffle(
            builder,
            options,
            text_encoder,
            pad_idx=pad_idx,
            max_seq_len=max_seq_len,
            text_column_name="text",
            device=gang.device,
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

        # Prepare full partition filters including split
        base_partition_filters = options.extras.get("partition_filters", None)
        full_partition_filters = self._prepare_full_partition_filters(
            partition_filters=base_partition_filters,
            split=split,
        )

        builder = ParquetTextDataset._build_pipeline_pre_batch_shuffle(
            text_encoder=text_encoder,
            pad_idx=pad_idx,
            gang=gang,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            options=options,
            parquet_files=self._dataset.files,  # type: ignore
            filesystem=self._dataset.filesystem,
            full_partition_filters=full_partition_filters,
        )

        pipeline = JsonlDataset.build_pipeline_post_batch_shuffle(builder, options)

        return DataPipelineReader[SequenceBatch](
            self._name, split or "default", pipeline, gang, options, strict_state=False
        )


class WeightedMixtureParquetDataset(ParquetTextDataset):

    def __init__(self, name: str, dataset: pq.ParquetDataset, splits: set[str]) -> None:
        super().__init__(name, dataset, splits)
        self._all_partitions_path_df = self._get_all_mixture_partitions()

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

        # Prepare full partition filters including split
        base_partition_filters = options.extras.get("partition_filters", None)
        full_partition_filters = self._prepare_full_partition_filters(
            partition_filters=base_partition_filters,
            split=split,
        )

        filtered_paths_df = pl.from_arrow(
            self._get_filtered_paths(
                self._all_partitions_path_df, full_partition_filters
            )
        )
        assert isinstance(filtered_paths_df, pl.DataFrame)

        weights_path = options.extras.get("weights_path", None)
        assert isinstance(
            weights_path, str
        ), "weights_path must be set for WeightedMixtureParquetDataset"

        weights_df = pl.read_parquet(weights_path)

        # Get all columns except 'weight' from weights_df for grouping and joining
        grouping_columns = [col for col in weights_df.columns if col != "weight"]

        partition_groups = (
            filtered_paths_df.group_by(grouping_columns)
            .agg(pl.col(self.parquet_path_name))
            .join(weights_df, on=grouping_columns, how="inner")
        )

        options.npc = max(options.npc // len(partition_groups), 1)
        options.num_prefetch = max(options.num_prefetch // len(partition_groups), 0)

        def _build_partition_pipeline(
            partition_group: Dict[str, Any]
        ) -> tuple[DataPipeline, float]:
            """Helper function to build pipeline for a single partition group."""
            mono_builder = self._build_pipeline_pre_batch_shuffle(
                text_encoder=text_encoder,
                pad_idx=pad_idx,
                gang=gang,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                options=options,
                parquet_files=partition_group[self.parquet_path_name],  # type: ignore
                filesystem=self._dataset.filesystem,
                full_partition_filters=None,
            )
            mono_pipeline = mono_builder.and_return()
            _ = next(iter(mono_pipeline))  # warmup
            return mono_pipeline, partition_group["weight"]

        pipelines: List[DataPipeline] = []
        weights: List[float] = []

        # Get max workers from options, default to number of partition groups (capped at 8)
        max_workers_raw = min(options.npc, len(partition_groups), 8)
        max_workers = (
            int(max_workers_raw)
            if max_workers_raw is not None
            else min(len(partition_groups), 8)
        )

        # Use ThreadPoolExecutor for parallel processing
        partition_groups_list = partition_groups.to_dicts()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_partition = {
                executor.submit(
                    _build_partition_pipeline, partition_group
                ): partition_group
                for partition_group in partition_groups_list
            }

            # Process completed tasks with progress bar
            for future in tqdm(
                as_completed(future_to_partition),
                total=len(future_to_partition),
                desc="Building partition pipelines",
            ):
                pipeline, weight = future.result()
                pipelines.append(pipeline)
                weights.append(weight)

        builder = DataPipeline.sample(
            pipelines=pipelines, weights=weights, seed=options.seed + gang.rank
        )

        pipeline = JsonlDataset.build_pipeline_post_batch_shuffle(builder, options)

        return DataPipelineReader[SequenceBatch](
            self._name, split or "default", pipeline, gang, options, strict_state=False
        )
