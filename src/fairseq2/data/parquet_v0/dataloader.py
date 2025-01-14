# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Generator
from functools import partial

from fairseq2.data import DataPipeline, DataPipelineBuilder
from fairseq2.data.parquet import (
    BatchOutputType,
    ParquetBasicDataloaderConfig,
    ParquetBatchFormat,
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
            filters=config.filters,  # type: ignore[arg-type]
            columns=config.columns,
            split_to_row_groups=config.split_to_row_groups,
            filesystem=config.filesystem,
            shuffle_window=(
                2 * config.nb_prefetch * config.nb_parallel_fragments
                if config.shuffle
                else None
            ),
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
        .bucket(config.nb_parallel_fragments)
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
) -> Generator[BatchOutputType, None, None]:
    """
    Example of usage:

       >>> from recipes.parquet.parquet_dataloader import (
       ...    ParquetBasicDataloaderConfig, ParquetBatchFormat, build_parquet_iterator_pipeline)
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
