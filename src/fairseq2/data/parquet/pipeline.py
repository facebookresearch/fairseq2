# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import partial
from pickle import dumps, loads
from typing import Generator, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from retrying import retry

from fairseq2.data import DataPipeline, DataPipelineBuilder, read_sequence
from fairseq2.data.parquet.configs import (
    ParquetBasicDataloaderConfig,
    ParquetBatchFormat,
    ParquetDatasetConfig,
    ParquetDatasetLimitOptions,
)
from fairseq2.data.parquet.transform import (
    add_fragments_trace,
    apply_filter,
    concat_table,
)
from fairseq2.data.parquet.utils import (
    BatchOutputType,
    add_partitioning_values,
    compute_length_splits,
    compute_rows_length,
    get_dataset_fragments,
    load_one_fragment,
    pyarrow_cpu,
    pyarrow_table_to_torch_dict,
    split_fragment_in_row_groups,
)
from fairseq2.logging import log

# --- tested above --- #


loading_retry = retry(
    retry_on_exception=lambda exception: isinstance(exception, OSError),
    stop_max_attempt_number=1,
    wait_exponential_multiplier=2,
    wait_exponential_max=20,
)


def init_parquet_dataset(
    parquet_path: str,
    filters: Optional[pa.dataset.Expression] = None,
) -> pq.ParquetDataset:
    """
    Initialize a Parquet dataset.
    Leaving `filesystem` to None will trigger the detection of the filesystem.

    Args:
        parquet_path (str): The path to the Parquet dataset.
        filters (Optional[pa.dataset.Expression]): Filters to apply to the dataset.

    Returns:
        pq.ParquetDataset: The initialized Parquet dataset.
    """
    return pq.ParquetDataset(parquet_path, filters=filters, filesystem=None)


class SafeFragment:
    """
    Simple wrapper around `ParquetFileFragment` that allows to reinit the state of filesystem
    if aws session token has expired.
    """

    fragment: pa.dataset.ParquetFileFragment

    def __init__(self, fragment: pa.dataset.ParquetFileFragment):
        self.fragment = fragment

    def __repr__(self) -> str:
        out = ""
        out += "SafeFragment \n"
        out += "path = " + self.fragment.path + "\n"
        out += f"row_groups = {[int(rg.id) for rg in self.fragment.row_groups]} \n"
        out += f"physical_schema = \n {self.fragment.physical_schema} \n"
        return out

    @loading_retry
    def load(
        self, columns: Optional[List[str]] = None, use_threads: bool = False
    ) -> pa.Table:
        if columns is not None:
            fragment_columns = [
                col for col in columns if col in self.fragment.physical_schema.names
            ]
        else:
            fragment_columns = self.fragment.physical_schema.names
        # adding technical columns for tracking
        fragment_columns = list(fragment_columns) + [
            "__batch_index",
            "__fragment_index",
            "__filename",
        ]
        try:
            fragment_table = self.fragment.to_table(
                columns=fragment_columns, use_threads=use_threads
            )

        except OSError as e:
            log.info(
                "could not load fragment, reinit the fragment state. Error: ", str(e)
            )
            self.fragment = loads(dumps(self.fragment))
            fragment_table = self.fragment.to_table(
                columns=fragment_columns, use_threads=use_threads
            )

        fragment_table = add_partitioning_values(fragment_table, self.fragment, columns)
        fragment_table = add_fragments_trace(fragment_table, self.fragment)
        return fragment_table


def list_parquet_fragments(
    parquet_path: str,
    filters: Optional[pa.dataset.Expression] = None,
    split_to_row_groups: bool = True,
    shuffle_window: Optional[int] = None,
    seed: int = 2,
    limit_options: Optional[ParquetDatasetLimitOptions] = None,
) -> DataPipelineBuilder:
    dataset = init_parquet_dataset(parquet_path, filters=filters)
    if limit_options:
        columns = limit_options.columns or dataset.schema.names
        if columns and not set(columns).issubset(set(dataset.schema.names)):
            raise ValueError(
                f"columns {sorted(set(columns) - set(dataset.schema.names))} are not found in the dataset schema"
            )

    pipeline_builder = read_sequence(get_dataset_fragments(dataset, filters))

    if shuffle_window is not None:
        # shuffle them in full memory since fragments are already known
        pipeline_builder = pipeline_builder.shuffle(shuffle_window=0, seed=seed)

    if split_to_row_groups:
        pipeline_builder = pipeline_builder.yield_from(
            lambda fragment: read_sequence(
                split_fragment_in_row_groups(fragment)
            ).and_return()
        )
        if shuffle_window is not None:
            pipeline_builder = pipeline_builder.shuffle(
                shuffle_window=shuffle_window, seed=seed + 1
            )

    return pipeline_builder


def build_iterator_over_one_table(
    table: pa.Table,
    order_by_length: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    num_parallel_calls: int = 8,
) -> DataPipeline:
    random_state = np.random.RandomState(seed)
    if order_by_length is not None:
        length_col = compute_rows_length(table[order_by_length])
        # add small perturbation to avoid same sample appear together during different epochs
        if shuffle:
            perturbation = random_state.randint(
                0,
                np.quantile(length_col, 0.001).astype(np.int32) + 2,
                len(length_col),
            )
            length_col += np.asarray(perturbation, dtype=np.int32)
    else:
        if shuffle:
            length_col = random_state.randint(0, 2**23, len(table))
        else:
            length_col = np.zeros(len(table), dtype=np.int32)

    if batch_size is not None:
        order_tt = pa.Table.from_arrays(
            [pa.array(np.argsort(length_col, kind="stable"))], ["order"]
        )
        batches = [ind["order"] for ind in order_tt.to_batches(batch_size)]
    elif max_tokens is not None:
        batches = compute_length_splits(length_col, max_tokens)
    else:
        raise ValueError("unknown batching method")

    if shuffle:
        batches = [batches[i] for i in random_state.permutation(len(batches))]

    return (
        read_sequence(batches)
        .map(
            lambda ind: table.take(ind).combine_chunks(),
            num_parallel_calls=num_parallel_calls,
        )
        .and_return(max_num_warnings=4)
    )


def build_parquet_iterator_pipeline(
    dataset_config: ParquetDatasetConfig,
    dataloader_config: ParquetBasicDataloaderConfig,
) -> DataPipelineBuilder:
    """Build a pipeline that iterates over parquet data with given configurations.

    Args:
        dataset_config: Configuration for the parquet dataset
        dataloader_config: Configuration for data loading behavior

    Returns:
        A DataPipelineBuilder that can be used to create the final pipeline
    """

    def inner_iterator(table: pa.Table) -> DataPipeline:
        """Creates an iterator over a single table with batching and ordering."""
        return build_iterator_over_one_table(
            table=table,
            order_by_length=dataloader_config.order_by_length,
            batch_size=dataloader_config.batch_size,
            max_tokens=dataloader_config.max_tokens,
            shuffle=dataloader_config.shuffle,
            seed=dataloader_config.seed,
            num_parallel_calls=max(dataloader_config.num_parallel_calls // 2, 1),
        )

    # Calculate shuffle window if needed
    shuffle_window = None
    if dataloader_config.shuffle:
        shuffle_window = (
            2 * dataloader_config.nb_prefetch * dataset_config.nb_parallel_fragments
        )

    # Build the main pipeline
    pipeline_builder = (
        list_parquet_fragments(
            parquet_path=dataset_config.parquet_path,
            filters=dataset_config.filters,  # type: ignore[arg-type]
            split_to_row_groups=dataset_config.split_to_row_groups,
            shuffle_window=shuffle_window,
            seed=dataloader_config.seed,
            limit_options=dataset_config.limit,
        )
        .shard(
            shard_idx=dataloader_config.rank,
            num_shards=dataloader_config.world_size,
        )
        .map(
            partial(
                load_one_fragment,
                columns=(
                    dataset_config.limit.columns if dataset_config.limit else None
                ),
            ),
            num_parallel_calls=dataloader_config.num_parallel_calls,
        )
    )

    # Apply filters if specified
    if dataset_config.filters is not None or dataloader_config.drop_null:
        pipeline_builder = pipeline_builder.map(
            partial(
                apply_filter,
                filters=dataset_config.filters,  # type: ignore[arg-type]
                drop_null=dataloader_config.drop_null,
            )
        )

    # Add bucketing and prefetch
    pipeline_builder = (
        pipeline_builder.bucket(dataset_config.nb_parallel_fragments)
        .prefetch(dataloader_config.nb_prefetch)
        .map(
            concat_table,
            num_parallel_calls=dataloader_config.nb_prefetch,
        )
        .yield_from(inner_iterator)
    )

    # Filter by minimum batch size if specified
    if dataloader_config.min_batch_size > 1:
        pipeline_builder = pipeline_builder.filter(
            lambda table: bool(len(table) >= dataloader_config.min_batch_size)
        )

    # Convert output format if needed
    if dataloader_config.output_format == ParquetBatchFormat.pandas:
        pipeline_builder = pipeline_builder.map(lambda table: table.to_pandas())
    elif dataloader_config.output_format == ParquetBatchFormat.torch:
        pipeline_builder = pipeline_builder.map(pyarrow_table_to_torch_dict)

    return pipeline_builder


def parquet_iterator(
    dataset_config: ParquetDatasetConfig,
    dataloader_config: ParquetBasicDataloaderConfig,
) -> Generator[BatchOutputType, None, None]:
    """
    Iterator over parquet data with given configurations.
    """
    with pyarrow_cpu(dataloader_config.num_parallel_calls):
        yield from iter(
            build_parquet_iterator_pipeline(dataset_config, dataloader_config)
            .prefetch(dataloader_config.num_parallel_calls)
            .and_return(max_num_warnings=4)
        )
