# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
]
