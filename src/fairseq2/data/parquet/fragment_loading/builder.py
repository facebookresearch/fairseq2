# Coeyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import gc
import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path
from pickle import dumps, loads
from typing import List, Optional

import numpy as np
import pyarrow as pa
from retrying import retry

from fairseq2.data import DataPipelineBuilder
from fairseq2.data.parquet.arrow_transform import (
    apply_filter,
)
from fairseq2.data.parquet.fragment_loading.config import FragmentLoadingConfig
from fairseq2.data.parquet.fragment_streaming.primitives import process_filter
from fairseq2.data.parquet.utils import (
    add_fragments_trace,
    add_partitioning_values,
    fragment_stable_hash,
    rename_table_columns,
    table_to_mmap_table,
)
from fairseq2.logging import log

loading_retry = retry(
    retry_on_exception=lambda exception: isinstance(exception, OSError),
    stop_max_attempt_number=1,
    wait_exponential_multiplier=2,
    wait_exponential_max=20,
)


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

    def stable_hash(self, seed=None) -> int:
        return fragment_stable_hash(self.fragment, seed)

    @loading_retry
    def load(
        self,
        columns: Optional[List[str]] = None,
        filters: Optional[pa.dataset.Expression] = None,
        use_threads: bool = False,
        add_fragment_traces: bool = True,
        add_partitioning_columns: bool = True,
    ) -> pa.Table:
        physical_schema = self.fragment.physical_schema
        if columns is not None:
            fragment_columns = [col for col in columns if col in physical_schema.names]
        else:
            fragment_columns = list(physical_schema.names)
        # adding technical columns for tracking
        if add_fragment_traces:
            fragment_columns = list(fragment_columns) + [
                "__batch_index",
                "__fragment_index",
                "__filename",
            ]

        can_apply_on_phyiscal_schema = False
        if filters is not None and not add_fragment_traces:
            try:
                _ = physical_schema.empty_table().filter(filters)
                can_apply_on_phyiscal_schema = True
            except pa.ArrowInvalid:
                pass

        try:
            fragment_table = self.fragment.to_table(
                columns=fragment_columns,
                use_threads=use_threads,
                filter=filters if can_apply_on_phyiscal_schema else None,
            )
        except OSError as e:
            log.info(
                "could not load fragment, reinit the fragment state. Error: ", str(e)
            )
            self.fragment = loads(dumps(self.fragment))
            fragment_table = self.fragment.to_table(
                columns=fragment_columns,
                use_threads=use_threads,
                filter=filters if can_apply_on_phyiscal_schema else None,
            )

        if add_partitioning_columns:
            fragment_table = add_partitioning_values(
                fragment_table, self.fragment, columns
            )
        if add_fragment_traces:
            fragment_table = add_fragments_trace(fragment_table, self.fragment)

        # otherwise, apply filters on full schema
        if filters is not None and not can_apply_on_phyiscal_schema:
            fragment_table = fragment_table.filter(filters)
        return fragment_table


class ParquetFragmentLoader:
    """This class is responsible for loading the fragments from the parquet files.
    It should be applied after the `ParquetFragmentStreamer !

    For better performance, use `with pyarrow_cpu(nb_cpus)` with big enough nb_cpus
    """

    def __init__(self, config: FragmentLoadingConfig) -> None:
        self.config: FragmentLoadingConfig = deepcopy(config)

        self.filters = process_filter(self.config.filters)

        if self.config.columns is not None:
            self.columns = self.config.columns.get_flatten_columns()
        else:
            self.columns = None

    def apply(self, fragment_pipeline: DataPipelineBuilder) -> DataPipelineBuilder:
        def load_fn(fragment: pa.dataset.ParquetFileFragment) -> pa.Table | None:
            safe_fragment = SafeFragment(fragment)
            if np.random.rand() < 0.05:
                gc.collect()
                # pa.jemalloc_set_decay_ms(10)
                pool = pa.default_memory_pool()
                pool.release_unused()

            table = safe_fragment.load(
                columns=self.columns,
                add_fragment_traces=self.config.add_fragment_traces,
                use_threads=self.config.use_threads,
                filters=self.filters,
                add_partitioning_columns=True,
            )
            return table

        if self.config.non_deterministic_read:
            # keeping if above checks for back-compatibility
            loading_pipeline = fragment_pipeline.map(
                load_fn,
                num_parallel_calls=self.config.num_parallel_fragments,
                deterministic=self.config.non_deterministic_read,
            )
        else:
            loading_pipeline = fragment_pipeline.map(
                load_fn,
                num_parallel_calls=self.config.num_parallel_fragments,
            )

        loading_pipeline = loading_pipeline.filter(
            lambda table: isinstance(table, pa.Table)
        )

        if self.config.drop_null:
            loading_pipeline = loading_pipeline.map(
                partial(
                    apply_filter,
                    filters=None,
                    drop_null=self.config.drop_null,
                )
            )

        loading_pipeline = loading_pipeline.filter(
            lambda table: bool(len(table) >= self.config.min_batch_size)
        )

        if self.config.columns is not None and self.config.rename_columns:
            loading_pipeline = loading_pipeline.map(
                partial(
                    rename_table_columns,
                    mapper=self.config.columns.get_renaming_mapper(),
                )
            )

        if self.config.cache:
            if self.config.cache_dir is not None:
                cache_dir = self.config.cache_dir
            else:
                cache_dir = tempfile.mkdtemp()

            log.info(
                f"Using cache dir = {cache_dir} to mmap the pa.Table after loading!"
            )
            # experiment !
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            loading_pipeline = loading_pipeline.map(
                partial(table_to_mmap_table, cache_dir=cache_dir)
            )

        loading_pipeline = loading_pipeline.prefetch(self.config.nb_prefetch)
        return loading_pipeline
