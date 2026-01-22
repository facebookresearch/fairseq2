# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa

from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder
from fairseq2.data.parquet.arrow_transform import concat_table, shuffle_table
from fairseq2.data.parquet.table_bucketing.config import TableBucketingConfig
from fairseq2.data.parquet.table_bucketing.primitives import (
    build_batching_loop_over_one_table,
    compute_rows_length,
)
from fairseq2.data.parquet.utils import table_to_mmap_table
from fairseq2.logging import log


class TableBucketer:
    def __init__(self, config: TableBucketingConfig):
        self.config = deepcopy(config)
        none_bucketing_params = [
            self.config.target_table_memory,
            self.config.target_total_length,
            self.config.target_table_size,
        ].count(None)

        if none_bucketing_params not in [2, 3]:
            raise ValueError(
                "Only one of `target_table_memory`, `target_total_length`, `target_table_size` can be set"
            )

        self.do_concat_tables = none_bucketing_params == 2

        if self.config.target_total_length is not None:
            assert (
                self.config.length_columns is not None
                and len(self.config.length_columns) > 0
            )

    def apply(self, pipeline: DataPipelineBuilder) -> DataPipelineBuilder:

        random_state = np.random.RandomState(self.config.seed)
        if self.config.target_table_memory is not None:

            def mem_mb(table: pa.Table) -> float:
                return table.nbytes / 1024**2

            pipeline = pipeline.dynamic_bucket(
                self.config.target_table_memory,
                mem_mb,
                min_num_examples=self.config.min_fragment_number,
                max_num_examples=self.config.max_fragment_number,
                drop_remainder=False,
            )
        elif self.config.target_table_size is not None:

            def size_fn(table: pa.Table) -> float:
                return len(table)

            pipeline = pipeline.dynamic_bucket(
                self.config.target_table_size,
                size_fn,
                min_num_examples=self.config.min_fragment_number,
                max_num_examples=self.config.max_fragment_number,
                drop_remainder=False,
            )
        elif self.config.target_total_length is not None:
            length_columns = self.config.length_columns
            assert length_columns is not None

            def len_fn(table: pa.Table) -> float:
                return sum(
                    compute_rows_length(table[column]).sum()
                    for column in length_columns
                    if column is not None
                )

            pipeline = pipeline.dynamic_bucket(
                self.config.target_total_length,
                len_fn,
                min_num_examples=self.config.min_fragment_number,
                max_num_examples=self.config.max_fragment_number,
                drop_remainder=False,
            )

        if self.do_concat_tables:
            pipeline = pipeline.map(
                lambda tables: concat_table(tables, combine=self.config.combine_chunks)
            )

        if (
            self.config.target_total_length is not None
            or self.config.batch_size is not None
            and self.config.shuffle
        ):
            # merely shuffle big concat table
            pipeline = pipeline.map(partial(shuffle_table, random_state=random_state))

        if self.config.cache:  # we cache big concat table !
            if self.config.cache_dir is not None:
                cache_dir = self.config.cache_dir
            else:
                cache_dir = tempfile.mkdtemp()

            log.info(
                f"Using cache dir = {cache_dir} to mmap the pa.Table after loading!"
            )
            # experiment !
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            pipeline = pipeline.map(partial(table_to_mmap_table, cache_dir=cache_dir))

        pipeline = pipeline.prefetch(self.config.nb_prefetch)
        # do some mini-batch dispatch
        if (
            self.config.total_batch_length is not None
            or self.config.batch_size is not None
        ):

            def mini_batch_iterator(table: pa.Table) -> DataPipeline:
                return build_batching_loop_over_one_table(
                    table=table,
                    order_by_length=self.config.order_by_length,
                    length_columns=self.config.length_columns,
                    batch_size=self.config.batch_size,
                    max_tokens=self.config.total_batch_length,
                    shuffle=self.config.shuffle,
                    seed=random_state.randint(0, 2**32),
                    len_reducer=self.config.length_reducer,
                    drop_long_sample=self.config.drop_long_seq,
                    num_parallel_calls=self.config.num_parallel_calls,
                )

            pipeline = pipeline.yield_from(mini_batch_iterator)

        return pipeline
