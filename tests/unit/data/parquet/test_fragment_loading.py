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


# @pytest.mark.parametrize("nb_epochs", [1, 2, 10])
# @pytest.mark.parametrize("shuffling_window", [-1, 0, 2, 4, 10])
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
        columns=CustomColumns(category="cat", uid="uid", extra_columns=["seq"])
    )
    PFL = ParquetFragmentLoader(config=loading_config)

    fragment_pipeline = PFS.build_pipeline(0, 1)
    loading_pipeline = PFL.build_pipeline(fragment_pipeline)

    result = list(iter(loading_pipeline.and_return()))

    total_number_of_row_groups = 6
    assert len(result) == 3 * total_number_of_row_groups
