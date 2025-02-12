# Coeyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc  # noqa: F401
import pyarrow.parquet as pq

from fairseq2.data import DataPipelineBuilder
from fairseq2.data.parquet.fragment_loading.config import FragmentLoadingConfig
from fairseq2.data.parquet.pipeline import SafeFragment
from fairseq2.data.parquet.transform import apply_filter
from fairseq2.data.parquet.utils import rename_table_columns, table_to_mmap_table
from fairseq2.logging import log


class ParquetFragmentLoader:
    """This class is responsible for loading the fragments from the parquet files.
    It should be applied after the `ParquetFragmentStreamer !

    For better performance, use `with pyarrow_cpu(nb_cpus)` with big enough nb_cpus
    """

    def __init__(self, config: FragmentLoadingConfig) -> None:
        self.config: FragmentLoadingConfig = deepcopy(config)
        if isinstance(self.config.filters, str):
            self.filters = pq.filters_to_expression(eval(self.config.filters))
        else:
            self.filters = self.config.filters

        if self.config.columns is not None:
            self.columns = self.config.columns.get_flatten_columns()
        else:
            self.columns = None

    def build_pipeline(
        self, fragment_pipeline: DataPipelineBuilder
    ) -> DataPipelineBuilder:
        def load_fn(safe_frag: SafeFragment) -> pa.Table | None:
            try:
                return safe_frag.load(
                    columns=self.columns,
                    add_fragment_traces=self.config.add_fragment_traces,
                    use_threads=self.config.use_threads,
                    add_partitioning_columns=True,
                )
            except Exception as e:
                log.error(
                    f"Error {e} occured while loading fragment {safe_frag} \n, skipping it"
                )
                return None

        # TODO: we want to do async loading of the fragments
        loading_pipeline = fragment_pipeline.map(
            load_fn,
            num_parallel_calls=self.config.num_parallel_fragments,
        )

        loading_pipeline = loading_pipeline.filter(
            lambda table: isinstance(table, pa.Table)
        )

        loading_pipeline = loading_pipeline.map(
            partial(
                apply_filter,
                filters=self.filters,
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
