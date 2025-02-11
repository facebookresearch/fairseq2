# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections import Counter
from dataclasses import dataclass
from typing import List

import numpy as np
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


def test_basic_fragment_loading(controled_row_groups_pq_dataset):
    fragment_config = FragmentStreamingConfig(
        parquet_path=controled_row_groups_pq_dataset,
        nb_epochs=3,
        seed=1,
        split_to_row_groups=True,
        fragment_shuffle_window=10,
        files_circular_shift=False,
    )
    PFS = ParquetFragmentStreamer(config=fragment_config)

    loading_config = FragmentLoadingConfig(
        columns=CustomColumns(category="cat", uid="id", extra_columns=["seq"]),
        cache=True,
        filters="pc.less_equal(pc.list_value_length(pc.field('seq')), 2)",
    )

    PFL = ParquetFragmentLoader(config=loading_config)

    fragment_pipeline = PFS.build_pipeline(0, 1)
    loading_pipeline = PFL.build_pipeline(fragment_pipeline)

    result = list(iter(loading_pipeline.and_return()))

    total_number_of_row_groups = 6
    assert len(result) == 3 * total_number_of_row_groups
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

    assert min(pc.list_value_length(x["seq"]).to_numpy().min() for x in result) == 2
