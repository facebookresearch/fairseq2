# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from functools import cached_property
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from fairseq2.data.data_pipeline import DataPipelineBuilder
from fairseq2.data.parquet.fragment_streaming.config import FragmentStreamingConfig
from fairseq2.data.parquet.fragment_streaming.primitives import (
    ParquetDatasetKey,
    ParquetDatasetWrapper,
    get_cached_dataset,
    get_dataset_cache_info,
    list_parquet_fragments,
    process_filter,
    stream_parquet_fragments,
)
from fairseq2.data.parquet.utils import fragment_stable_hash
from fairseq2.logging import log


class ParquetFragmentStreamer:
    """A streamer for parquet dataset fragments.

    This class manages the streaming of fragments from a parquet dataset. It uses
    a module-level singleton cache to avoid redundant instantiation of the underlying
    PyArrow Dataset when multiple streamer instances reference the same parquet path
    and filesystem.

    The cache key intentionally excludes partition_filters since the expensive
    ds.dataset() call doesn't depend on filters - filters are applied later when
    retrieving fragments.

    Args:
        config: Configuration for the fragment streaming.
        use_cache: Whether to use the singleton cache. Defaults to True.
            Set to False for testing or when cache isolation is needed.
    """

    def __init__(
        self, config: FragmentStreamingConfig, *, use_cache: bool = False
    ) -> None:
        self.config: FragmentStreamingConfig = deepcopy(config)
        self._pq_ds: Optional[ParquetDatasetWrapper] = None
        self._use_cache = use_cache

        if (
            self.config.files_circular_shift
            and self.config.fragment_shuffle_window == -1
        ):
            log.info(
                "Cannot use files circular shift and full shuffle at the same time. "
                "Ignoring files circular shift."
            )
            self.config.files_circular_shift = False

    @property
    def dataset(self) -> ParquetDatasetWrapper:
        if self._pq_ds is None:
            self._pq_ds = self._get_dataset()

        return self._pq_ds

    @cached_property
    def partition_filters(self) -> pc.Expression:
        return process_filter(self.config.partition_filters)

    def _get_dataset(self) -> ParquetDatasetWrapper:
        partition_filters = self.partition_filters
        if isinstance(self.config.filesystem, str):
            self.filesystem = eval(self.config.filesystem)
        else:
            self.filesystem = self.config.filesystem

        dataset_key = ParquetDatasetKey.from_init_args(
            self.config.parquet_path,
            self.filesystem,
        )

        if self._use_cache:
            cache_info_before = get_dataset_cache_info()
            cached_dataset = get_cached_dataset(dataset_key)
            cache_info_after = get_dataset_cache_info()
            if cache_info_after.misses == cache_info_before.misses:
                log.debug(f"Dataset cache hit for key: {dataset_key}")
        else:
            log.debug(
                f"Cache disabled, initializing new ds.Dataset for key: {dataset_key}"
            )
            cached_dataset = ds.dataset(
                (
                    dataset_key.path
                    if isinstance(dataset_key.path, str)
                    else list(dataset_key.path)
                ),
                format="parquet",
                partitioning="hive",
                filesystem=self.filesystem,
            )

        return ParquetDatasetWrapper(
            dataset=cached_dataset,
            partition_filters=partition_filters,
        )

    @property
    def full_schema(self) -> pa.Schema:
        return self.dataset.schema

    def build_pipeline(
        self, rank: int = 0, world_size: int = 1, even_sharding: bool = False
    ) -> DataPipelineBuilder:
        """
        Build a pipeline of parquet fragments and next will be shared to a given rank and world size.
        """
        if even_sharding and self.config.files_circular_shift:
            raise ValueError(
                "Cannot use even sharding and files circular shift at the same time"
            )

        shuffle = self.config.fragment_shuffle_window != 0

        if (self.config.fragment_shuffle_window == -1) or (
            self.config.limit is not None
            and (
                self.config.limit.nb_fragments is not None
                or self.config.limit.nb_rows is not None
            )
        ):
            # doing full scan of the dataset to shuffle row groups globally
            # this produces the same ordering for all ranks for a fixed seed
            fragments_pipeline_builder = list_parquet_fragments(
                parquet_ds=self.dataset,
                nb_epochs=self.config.nb_epochs,
                split_to_row_groups=self.config.split_to_row_groups,
                shuffle=shuffle,
                seed=self.config.seed,
                limit_options=self.config.limit,
            )
            fragments_pipeline_builder = fragments_pipeline_builder.shard(
                shard_idx=rank,
                num_shards=world_size,
                allow_uneven=not even_sharding,
            )
            return fragments_pipeline_builder

        files_circular_shift_amount = (
            rank / world_size if self.config.files_circular_shift else 0.0
        )
        fragments_pipeline_builder = stream_parquet_fragments(
            parquet_ds=self.dataset,
            nb_epochs=self.config.nb_epochs,
            split_to_row_groups=self.config.split_to_row_groups,
            shuffle=shuffle,
            seed=self.config.seed,
            limit_options=self.config.limit,
            shuffling_window=self.config.fragment_shuffle_window,
            files_circular_shift=files_circular_shift_amount,
        )

        if self.config.files_circular_shift:
            # this makes sure that each rank will get different set row groups for each epoch
            # whaterver internal shuffle is done
            fragments_pipeline_builder = fragments_pipeline_builder.filter(
                lambda fragment: fragment_stable_hash(fragment, seed=self.config.seed)
                % world_size
                == rank
            )
        else:
            fragments_pipeline_builder = fragments_pipeline_builder.shard(
                shard_idx=rank,
                num_shards=world_size,
                allow_uneven=not even_sharding,
            )

        return fragments_pipeline_builder
