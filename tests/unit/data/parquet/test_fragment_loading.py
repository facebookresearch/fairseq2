# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import List

import pyarrow as pa
import pyarrow.compute as pc
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


@dataclass
class CustomColumns(NamedColumns):
    category: str
    uid: str
    extra_columns: List[str]


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("nb_epochs", [1, 2, 4])
@pytest.mark.parametrize("seed", [10, 20, 40])
def test_basic_fragment_loading(
    controled_row_groups_pq_dataset, nb_epochs, cache, seed
):
    fragment_config = FragmentStreamingConfig(
        parquet_path=controled_row_groups_pq_dataset,
        nb_epochs=nb_epochs,
        seed=seed,
        split_to_row_groups=True,
        fragment_shuffle_window=10,
        files_circular_shift=False,
    )
    PFS = ParquetFragmentStreamer(config=fragment_config)

    loading_config = FragmentLoadingConfig(
        columns=CustomColumns(category="cat", uid="id", extra_columns=["seq"]),
        cache=cache,
        filters="pc.less_equal(pc.list_value_length(pc.field('seq')), 3)",
    )

    PFL = ParquetFragmentLoader(config=loading_config)

    fragment_pipeline = PFS.build_pipeline(0, 1)
    loading_pipeline = PFL.apply(fragment_pipeline)

    result = list(iter(loading_pipeline.and_return()))

    total_number_of_row_groups = 6
    assert len(result) == nb_epochs * total_number_of_row_groups
    assert all(isinstance(x, pa.Table) for x in result)
    assert all(len(x) >= 1 for x in result)
    expected_columns = [
        "uid",
        "seq",
        "__batch_index",
        "__fragment_index",
        "__filename",
        "category",
        "__row_groups_ids",
        "__index_in_fragement",
    ]
    assert all(sorted(x.column_names) == sorted(expected_columns) for x in result)

    assert max(pc.list_value_length(x["seq"]).to_numpy().max() for x in result) == 3


@pytest.mark.parametrize("nb_epochs", [1, 2, 4])
@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("seed", [10, 20, 40])
def test_basic2_fragment_loading(
    controled_row_groups_pq_dataset, nb_epochs, cache, seed
):
    fragment_config = FragmentStreamingConfig(
        parquet_path=controled_row_groups_pq_dataset,
        nb_epochs=nb_epochs,
        seed=seed,
        partition_filters="pc.greater_equal(pc.field('cat'), 'cat_1')",
        split_to_row_groups=True,
        fragment_shuffle_window=10,
        files_circular_shift=True,
    )
    PFS = ParquetFragmentStreamer(config=fragment_config)

    loading_config = FragmentLoadingConfig(
        columns=CustomColumns(category="cat", uid="id", extra_columns=["seq"]),
        cache=cache,
        rename_columns=False,
        add_fragment_traces=False,
        filters="pc.greater_equal(pc.list_value_length(pc.field('seq')), 4)",
    )

    PFL = ParquetFragmentLoader(config=loading_config)

    fragment_pipeline = PFS.build_pipeline(0, 1)
    loading_pipeline = PFL.apply(fragment_pipeline)

    result = list(iter(loading_pipeline.and_return()))

    total_number_of_row_groups = 6 - 2  # 2 row groups are filtered out in cat=0
    assert len(result) == nb_epochs * total_number_of_row_groups
    assert all(isinstance(x, pa.Table) for x in result)
    assert all(len(x) >= 1 for x in result)
    expected_columns = [
        "id",
        "seq",
        "cat",
    ]
    assert all(sorted(x.column_names) == sorted(expected_columns) for x in result)

    assert min(pc.list_value_length(x["seq"]).to_numpy().min() for x in result) == 4
