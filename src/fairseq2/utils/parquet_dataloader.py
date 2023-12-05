# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from dataclasses import dataclass
from enum import Enum
from functools import partial

import numpy as np
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
import torch

from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder, read_sequence
from fairseq2.utils.parquet_tools import (
    BatchOutputType,
    NestedDict,
    apply_filter,
    compute_length_splits,
    compute_rows_length,
    concat_table,
    get_dataset_fragments,
    init_parquet_dataset,
    load_one_fragment,
    pyarrow_cpu,
    pyarrow_table_to_torch_dict,
    split_fragment_in_row_groups,
)


class _TableWrapper:
    """
    class to avoid fairseq2 casting pa.Table to iterable objects
    which currently fails
    """

    def __init__(self, table: pa.Table) -> None:
        self.table: pa.Table = table


def _to_real_object(x: tp.Union[_TableWrapper, NestedDict]) -> BatchOutputType:
    if isinstance(x, _TableWrapper):
        return x.table
    elif isinstance(x, list):
        return [_to_real_object(e) for e in x]
    elif isinstance(x, tuple):
        return tuple(_to_real_object(e) for e in x)
    else:
        return x


def table_func_wrap(func):  # type: ignore
    def inner(*args):  # type: ignore
        fixed_args = [_to_real_object(x) for x in args]
        result = func(*fixed_args)
        if isinstance(result, (pa.Table, pd.DataFrame)):
            result = _TableWrapper(result)
        return result

    return inner


def build_iterator_over_one_table(
    table: pa.Table,
    order_by: tp.Optional[str] = None,
    batch_size: tp.Optional[int] = None,
    max_tokens: tp.Optional[int] = None,
    shuffle: bool = True,
    seed: tp.Optional[int] = None,
    num_parallel_calls: int = 8,
) -> DataPipeline:
    random_state = np.random.RandomState(seed)
    if order_by is not None:
        length_col = compute_rows_length(table[order_by])
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
            table_func_wrap(lambda ind: table.take(ind).combine_chunks()),
            num_parallel_calls=num_parallel_calls,
        )
        .and_return(max_num_warnings=4)
    )


class ParquetBatchFormat(Enum):
    pyarrow = 0
    pandas = 1
    torch = 2


@dataclass  # TODO: (kw_only=True) with python3.10
class ParquetBasicDataloaderConfig:
    parquet_path: str
    """
    Path to parquet dataset file
    """

    batch_size: tp.Optional[int] = None
    """
    Fixed output batch size
    """

    order_by: tp.Optional[str] = None
    """Column in dataset whose value length `L` will be used for batches ordering.
       This results in batches with relatively homogeneous values of `L`,
       typically to support optimal padding.
    """

    max_tokens: tp.Optional[int] = None
    """
    Used with `order_by` option to control the total number of padded tokens in a each batch.
    Typically, this option is preferred to `batch_size` for reducing the memory footprint.
    """

    columns: tp.Optional[tp.List[str]] = None
    """List of columns to load"""

    filters: tp.Optional[tp.Union[tp.List[tp.Any], pa.dataset.Expression]] = None
    """
    See https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html#pyarrow.dataset.Expression
    Some examples :

    >>> import pyarrow.compute as pc
    >>> import pyarrow as pa

    >>> filters = [("data_split", "=", "train"), ("lang1", "in", ["eng","spa"]), ("lang2", "=", "eng")])
    >>> filters = (pc.field("data_split") == pc.scalar("train")) & (pc.field("duration") > 7)
    >>> filters = pa.compute.greater(pa.compute.utf8_length(ds.field("lang1_text")), 4)
    >>> filters = pa.compute.less_equal(pa.compute.list_value_length(pa.dataset.field("audio_wav")), 16_000 * 30)

    Note that all fields used here should be among existing columns in the dataset schema
    """

    output_format: ParquetBatchFormat = ParquetBatchFormat.pyarrow
    """
    Format to use for output batches
    """

    split_to_row_groups: bool = True
    """
    Use parquet row groups instead of simple partitions which are generally smaller.
    Highly recommended for non-partitioned parquet file.
    """

    shuffle: bool = True
    """
    Whether to shuffle dataset samples during the iteration.
    If False and `order_by` is None, the batch samples will be produced in natural parquet dataset reading order.
    """

    drop_null: bool = True
    """Dropping rows containing any null value"""

    seed: tp.Optional[int] = None
    """
    seed making iteration deterministic
    """

    min_batch_size: int = 1
    """Drops batches whose length < `min_batch_size`"""

    nb_producers: int = 5
    """Number of parquet partitions read allowed to be read synonymously.
       Higher values will result in higher speed, better randomization and higher memory footprint.
       If partitions size is rather small compared to batch size, we recommend to increase nb_producers.
    """

    nb_prefetch: int = 2
    """
    Nb of producers groups (of size `nb_producers`) to prefetch
    """

    rank: int = 0
    """The rank of this worker in the process group."""

    world_size: int = 1
    """The world size of the process group."""

    num_parallel_calls: int = 8
    """The number of parallel calls in map operations."""

    use_threads: bool = False
    """
    Whether pyarrow should use its internal parallelism threads to read the parquet part.
    Since we rely on the external parallelism, this param is tuned off.
    """
    filesystem: tp.Optional[pa.fs.FileSystem] = None
    """
    Filesystem to read parquet files from. S3 example :
    >>> import s3fs
    >>> filesystem = s3fs.core.S3FileSystem(...)
    """

    def __post_init__(self) -> None:
        assert self.parquet_path, "requires path"
        assert self.num_parallel_calls >= 1
        assert self.world_size >= 1
        assert self.world_size - 1 >= self.rank >= 0
        assert self.nb_prefetch >= 1
        assert self.nb_producers >= 1

        assert self.min_batch_size >= 0
        assert self.batch_size is None or self.batch_size > 0
        assert self.max_tokens is None or self.max_tokens > 0

        if not ((self.batch_size is None) ^ (self.max_tokens is None)):
            raise ValueError("need to provide either `batch_size` either `max_tokens`")
        if self.max_tokens is not None and self.order_by is None:
            raise ValueError("`order_by` should be given to deal with `max_tokens`")

        if self.filters is not None and not isinstance(
            self.filters, pa.dataset.Expression
        ):
            self.filters = pq.filters_to_expression(self.filters)


def build_parquet_iterator_pipeline(
    config: ParquetBasicDataloaderConfig,
) -> DataPipelineBuilder:
    seed = config.seed
    if seed:
        torch.manual_seed(seed + 123)  # used by `DataPipeline.shuffle`

    dataset = init_parquet_dataset(
        config.parquet_path, filters=config.filters, filesystem=config.filesystem
    )
    columns = config.columns or dataset.schema.names
    assert set(columns).issubset(set(dataset.schema.names))

    def inner_iterator(wrap_table: _TableWrapper) -> DataPipeline:
        return build_iterator_over_one_table(
            table=wrap_table.table,
            order_by=config.order_by,
            batch_size=config.batch_size,
            max_tokens=config.max_tokens,
            shuffle=config.shuffle,
            seed=seed,
            num_parallel_calls=max(config.num_parallel_calls // 2, 1),
        )

    pipeline_builder = read_sequence(get_dataset_fragments(dataset, config.filters))

    if config.shuffle:
        # shuffle them in full memory since fragments are already known
        pipeline_builder = pipeline_builder.shuffle(shuffle_window=0)

    if config.split_to_row_groups:
        pipeline_builder = pipeline_builder.yield_from(
            lambda fragment: read_sequence(
                split_fragment_in_row_groups(fragment)
            ).and_return()
        )
    if config.shuffle:
        pipeline_builder = pipeline_builder.shuffle(
            shuffle_window=2 * config.nb_prefetch * config.nb_producers
        )

    pipeline_builder = (
        pipeline_builder.shard(shard_idx=config.rank, num_shards=config.world_size)
        .map(
            table_func_wrap(partial(load_one_fragment, columns=columns)),
            num_parallel_calls=config.num_parallel_calls,
        )
        .map(
            table_func_wrap(
                partial(
                    apply_filter, filters=config.filters, drop_null=config.drop_null
                )
            )
        )
        .bucket(config.nb_producers)
        .prefetch(config.nb_prefetch)
        .map(
            table_func_wrap(concat_table),
            num_parallel_calls=config.nb_prefetch,
        )
        .yield_from(inner_iterator)
        .filter(
            table_func_wrap(lambda table: bool(len(table) >= config.min_batch_size))
        )
    )

    if config.output_format == ParquetBatchFormat.pandas:
        pipeline_builder = pipeline_builder.map(
            table_func_wrap(lambda table: table.to_pandas())
        )
    elif config.output_format == ParquetBatchFormat.torch:
        pipeline_builder = pipeline_builder.map(
            table_func_wrap(pyarrow_table_to_torch_dict)
        )
    return pipeline_builder


def parquet_iterator(
    config: ParquetBasicDataloaderConfig,
) -> tp.Generator[BatchOutputType, None, None]:
    """
    Example of usage :

       >>> from fairseq2.utils.parquet_dataloader import ParquetBasicDataloaderConfig, parquet_iterator
       >>> from tqdm.auto import tqdm
       >>> bpd_config = ParquetBasicDataloaderConfig(parquet_path="...", batch_size=20,
       ...                                           columns=["src_text", "src_lang", "audio_wav"],
       ...                                           output_format=ParquetBatchFormat.torch)
       >>> ei_batch = parquet_iterator(bpd_config)
       >>> res = []
       >>> for i, batch in tqdm(enumerate(ei_batch)): res.append(len(batch))
    """
    with pyarrow_cpu(config.num_parallel_calls):
        yield from map(
            _to_real_object,
            iter(
                build_parquet_iterator_pipeline(config)
                .prefetch(config.num_parallel_calls)
                .and_return(max_num_warnings=4)
            ),
        )
