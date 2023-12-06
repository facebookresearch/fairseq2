# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from dataclasses import dataclass
from enum import Enum
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq

from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder
from fairseq2.utils.parquet_tools import (
    BatchOutputType,
    _TableWrapper,
    _to_real_object,
    apply_filter,
    build_iterator_over_one_table,
    concat_table,
    list_parquet_fragments,
    load_one_fragment,
    pyarrow_cpu,
    pyarrow_table_to_torch_dict,
    table_func_wrap,
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

    order_by_length: tp.Optional[str] = None
    """Column in dataset whose value length `L` will be used for batches ordering.
       This results in batches with relatively homogeneous values of `L`,
       typically to support optimal padding.
    """

    max_tokens: tp.Optional[int] = None
    """
    Used with `order_by_length` option to control the total number of padded tokens in a each batch.
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
    If False and `order_by_length` is None, the batch samples will be produced in natural parquet dataset reading order.
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
        if self.max_tokens is not None and self.order_by_length is None:
            raise ValueError(
                "`order_by_length` should be given to deal with `max_tokens`"
            )

        if self.filters is not None and not isinstance(
            self.filters, pa.dataset.Expression
        ):
            self.filters = pq.filters_to_expression(self.filters)


def build_parquet_iterator_pipeline(
    config: ParquetBasicDataloaderConfig,
) -> DataPipelineBuilder:
    def inner_iterator(wrap_table: _TableWrapper) -> DataPipeline:
        return build_iterator_over_one_table(
            table=wrap_table.table,
            order_by_length=config.order_by_length,
            batch_size=config.batch_size,
            max_tokens=config.max_tokens,
            shuffle=config.shuffle,
            seed=config.seed,
            num_parallel_calls=max(config.num_parallel_calls // 2, 1),
        )

    pipeline_builder = (
        list_parquet_fragments(
            parquet_path=config.parquet_path,
            filters=config.filters,
            columns=config.columns,
            split_to_row_groups=config.split_to_row_groups,
            filesystem=config.filesystem,
            shuffle_window=2 * config.nb_prefetch * config.nb_producers
            if config.shuffle
            else None,
            seed=config.seed,
        )
        .shard(shard_idx=config.rank, num_shards=config.world_size)
        .map(
            table_func_wrap(partial(load_one_fragment, columns=config.columns)),
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
