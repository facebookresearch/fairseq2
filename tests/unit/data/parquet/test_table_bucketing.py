# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections import Counter
from dataclasses import dataclass
from typing import List

import pyarrow as pa
import pytest

from fairseq2.data.parquet.fragment_loading import (
    FragmentLoadingConfig,
    NamedColumns,
    ParquetFragmentLoader,
)
from fairseq2.data.parquet.fragment_streaming import (
    FragmentStreamingConfig,
    ParquetFragmentStreamer,
)
from fairseq2.data.parquet.table_bucketing import TableBucketer, TableBucketingConfig


@dataclass
class CustomColumns(NamedColumns):
    category: str
    uid: str
    extra_columns: List[str]


@pytest.mark.parametrize("nb_epochs", [10])
def test_basic_bucketing(controled_row_groups_pq_dataset, nb_epochs):
    fragment_config = FragmentStreamingConfig(
        parquet_path=controled_row_groups_pq_dataset,
        nb_epochs=nb_epochs,
        seed=123,
        split_to_row_groups=True,
        fragment_shuffle_window=0,  # no shuffling
        files_circular_shift=False,
    )

    loading_config = FragmentLoadingConfig(
        columns=CustomColumns(category="cat", uid="id", extra_columns=["seq"]),
        add_fragment_traces=False,
    )

    PFS = ParquetFragmentStreamer(config=fragment_config)
    PFL = ParquetFragmentLoader(config=loading_config)

    bucketing_config = TableBucketingConfig(target_table_size=100, shuffle=False)
    empty_bucketing_config = TableBucketingConfig()

    def get_all_tables(bucketing_config: TableBucketingConfig):
        fragment_pipeline = PFS.build_pipeline(0, 1)
        loading_pipeline = PFL.apply(fragment_pipeline)
        bucketing_pipeline = TableBucketer(bucketing_config).apply(loading_pipeline)
        return list(iter(bucketing_pipeline.and_return()))

    no_bucketing_result = get_all_tables(empty_bucketing_config)

    total_number_of_row_groups = 6
    assert len(no_bucketing_result) == nb_epochs * total_number_of_row_groups
    assert all(isinstance(x, pa.Table) for x in no_bucketing_result)
    assert all(len(x) >= 1 for x in no_bucketing_result)
    expected_columns = [
        "uid",
        "seq",
        "category",
    ]
    assert all(
        sorted(x.column_names) == sorted(expected_columns) for x in no_bucketing_result
    )

    # no let compare with bucketing
    bucketing_result = get_all_tables(bucketing_config)

    assert all(isinstance(x, pa.Table) for x in bucketing_result)
    assert all(
        sorted(x.column_names) == sorted(expected_columns) for x in bucketing_result
    )
    assert all(len(x) >= 1 for x in bucketing_result)

    assert len(bucketing_result) == nb_epochs * total_number_of_row_groups // 10
    assert Counter(len(x) for x in bucketing_result) == Counter({100: 6})

    # check that id are contiguous
    full_table = pa.concat_tables(bucketing_result)
    assert (
        full_table["uid"].to_pylist()
        == pa.concat_tables(no_bucketing_result)["uid"].to_pylist()
    )

    # dispatch in mini batches
    mini_bs_bucketing_config = TableBucketingConfig(
        target_table_size=151, batch_size=5, shuffle=False
    )
    mini_bucketing_result = get_all_tables(mini_bs_bucketing_config)

    assert all(isinstance(x, pa.Table) for x in mini_bucketing_result)
    assert all(
        sorted(x.column_names) == sorted(expected_columns)
        for x in mini_bucketing_result
    )

    assert len(mini_bucketing_result) == nb_epochs * total_number_of_row_groups * 2
    assert Counter(len(x) for x in mini_bucketing_result) == Counter({5: 120})

    # check that id are contiguous
    assert (
        pa.concat_tables(mini_bucketing_result)["uid"].to_pylist()
        == pa.concat_tables(no_bucketing_result)["uid"].to_pylist()
    )
