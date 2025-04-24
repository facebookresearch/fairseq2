# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq2.data.data_pipeline import DataPipelineBuilder
from fairseq2.data.parquet.arrow_transform import (
    apply_filter,
    concat_table,
    filter_list_by_range,
    filter_strings_by_length,
    maybe_cast,
    repeat_list_column,
    replace_table_column,
)
from fairseq2.data.parquet.fragment_loading import (
    FragmentLoadingConfig,
    NamedColumns,
    ParquetFragmentLoader,
)
from fairseq2.data.parquet.fragment_streaming import (
    FragmentStreamingConfig,
    ParquetFragmentStreamer,
    RejectionDistributionSmoother,
)
from fairseq2.data.parquet.fragment_streaming.primitives import (
    init_parquet_dataset,
    stream_parquet_fragments,
)
from fairseq2.data.parquet.table_bucketing import TableBucketer, TableBucketingConfig
from fairseq2.data.parquet.utils import (
    BatchOutputType,
    NestedDict,
    NestedDictValue,
    get_dataset_fragments,
    pyarrow_column_to_array,
    pyarrow_cpu,
    pyarrow_table_to_torch_dict,
    pyarrow_to_torch_tensor,
    split_fragment_in_row_groups,
)


@dataclass
class BasicDataLoadingConfig:
    fragment_stream_config: FragmentStreamingConfig

    # default trivial config will load all columns
    fragment_load_config: FragmentLoadingConfig = field(
        default_factory=lambda: FragmentLoadingConfig()
    )

    # default trivial config applies NO bucketing
    table_bucketing_config: TableBucketingConfig = field(
        default_factory=lambda: TableBucketingConfig()
    )


def build_basic_parquet_data_pipeline(
    config: BasicDataLoadingConfig, rank: int = 0, world_size: int = 1
) -> DataPipelineBuilder:
    """
    Simple integration of Parquet components to build a basic dataloading pipeline.

    >>> from fairseq2.data.parquet import *
    >>> config = BasicDataLoadingConfig(
    ...     fragment_stream_config=FragmentStreamingConfig(
    ...         parquet_path="path/to/parquet",
    ...         partition_filters='pc.field("split") == "train"',
    ...         nb_epochs=None,
    ...         fragment_shuffle_window=100),
    ...     fragment_load_config=FragmentLoadingConfig(columns=None, nb_prefetch=2, num_parallel_fragments=3),
    ... )
    >>> pipeline = build_basic_parquet_data_pipeline(config).and_return()
    >>> for batch in pipeline:
    ...     print(batch.to_pandas())

    """

    pipeline = ParquetFragmentStreamer(
        config=config.fragment_stream_config
    ).build_pipeline(rank, world_size)
    pipeline = ParquetFragmentLoader(config=config.fragment_load_config).apply(pipeline)
    pipeline = TableBucketer(config=config.table_bucketing_config).apply(pipeline)

    return pipeline


__all__ = [
    "BatchOutputType",
    "NestedDict",
    "NestedDictValue",
    "get_dataset_fragments",
    "pyarrow_cpu",
    "pyarrow_table_to_torch_dict",
    "pyarrow_to_torch_tensor",
    "pyarrow_column_to_array",
    "split_fragment_in_row_groups",
    # --- transform --- #
    "apply_filter",
    "concat_table",
    "replace_table_column",
    "repeat_list_column",
    "filter_list_by_range",
    "filter_strings_by_length",
    "maybe_cast",
    # --- pipeline --- #
    "init_parquet_dataset",
    "stream_parquet_fragments",
    "FragmentLoadingConfig",
    "NamedColumns",
    "ParquetFragmentLoader",
    "FragmentStreamingConfig",
    "ParquetFragmentStreamer",
    "TableBucketer",
    "TableBucketingConfig",
    "BasicDataLoadingConfig",
    "build_basic_parquet_data_pipeline",
]
