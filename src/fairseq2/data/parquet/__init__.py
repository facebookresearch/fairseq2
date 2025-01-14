# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.data.parquet.configs import (
    DataLoadingConfig,
    EvaluationDataLoadingConfig,
    ParquetBasicDataloaderConfig,
    ParquetBatchFormat,
    ParquetDatasetLimitOptions,
    ValidationDataLoadingConfig,
)
from fairseq2.data.parquet.pipeline import (
    build_iterator_over_one_table,
    init_parquet_dataset,
    list_parquet_fragments,
    parquet_iterator,
)
from fairseq2.data.parquet.transform import (
    affix_list_column,
    apply_filter,
    build_uniform_list_column,
    concat_table,
    filter_list_by_range,
    filter_rows_by_consistent_list_length,
    filter_strings_by_length,
    maybe_cast,
    replace_table_column,
)
from fairseq2.data.parquet.utils import (
    BatchOutputType,
    NestedDict,
    NestedDictValue,
    _TableWrapper,
    _to_real_object,
    add_partitioning_values,
    compute_length_splits,
    compute_rows_length,
    get_dataset_fragments,
    hstack_pyarray_list,
    load_one_fragment,
    pyarrow_column_to_array,
    pyarrow_cpu,
    pyarrow_table_to_torch_dict,
    pyarrow_to_torch_tensor,
    split_fragment_in_row_groups,
    table_func_wrap,
    torch_random_seed,
)

__all__ = [
    # --- configs --- #
    "ParquetDatasetLimitOptions",
    "DataLoadingConfig",
    "ValidationDataLoadingConfig",
    "EvaluationDataLoadingConfig",
    "ParquetBatchFormat",
    "ParquetBasicDataloaderConfig",
    # --- utils --- #
    "_TableWrapper",
    "_to_real_object",
    "BatchOutputType",
    "NestedDict",
    "NestedDictValue",
    "add_partitioning_values",
    "compute_length_splits",
    "compute_rows_length",
    "get_dataset_fragments",
    "hstack_pyarray_list",
    "load_one_fragment",
    "pyarrow_cpu",
    "pyarrow_table_to_torch_dict",
    "pyarrow_to_torch_tensor",
    "pyarrow_column_to_array",
    "split_fragment_in_row_groups",
    "table_func_wrap",
    "torch_random_seed",
    # --- transform --- #
    "apply_filter",
    "concat_table",
    "replace_table_column",
    "affix_list_column",
    "build_uniform_list_column",
    "filter_list_by_range",
    "filter_rows_by_consistent_list_length",
    "filter_strings_by_length",
    "maybe_cast",
    # --- pipeline --- #
    "init_parquet_dataset",
    "list_parquet_fragments",
    "build_iterator_over_one_table",
    "parquet_iterator",
]
