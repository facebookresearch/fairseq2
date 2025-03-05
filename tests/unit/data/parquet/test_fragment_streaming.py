# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fairseq2.data.parquet.fragment_streaming.basic_pipeline import (
    list_parquet_fragments,
    stream_parquet_fragments,
)
from fairseq2.data.parquet.fragment_streaming.builder import ParquetFragmentStreamer
from fairseq2.data.parquet.fragment_streaming.config import (
    FragmentStreamingConfig,
    ParquetDatasetLimitOptions,
)


def are_fragments_equal(fragment1, fragment2):
    if fragment1.path != fragment2.path:
        return False

    if fragment1.metadata.to_dict() != fragment2.metadata.to_dict():
        return False

    if fragment1.physical_schema != fragment2.physical_schema:
        return False
    if [int(rg.id) for rg in fragment1.row_groups] != [
        int(rg.id) for rg in fragment2.row_groups
    ]:
        return False

    # # Optionally, compare the data
    # table1 = fragment1.to_table()
    # table2 = fragment2.to_table()
    # if not table1.equals(table2):
    #     return False

    return True


@pytest.mark.parametrize("nb_epochs", [1, 2, 10])
@pytest.mark.parametrize("shuffling_window", [1, 2, 4, 10])
@pytest.mark.parametrize("files_circular_shift", [0, 0.334, 0.5, 1.0])
@pytest.mark.parametrize("stop_id", [5, 15, 20])
@pytest.mark.parametrize("shuffle", [True, False])
def test_stream_parquet_fragments_generic(
    controled_row_groups_pq_dataset,
    nb_epochs: int,
    shuffling_window: int,
    files_circular_shift: float,
    shuffle: bool,
    stop_id: int,
):
    parquet_ds = pq.ParquetDataset(controled_row_groups_pq_dataset)
    total_number_of_row_groups = 6

    def get_new_stream():
        return stream_parquet_fragments(
            parquet_ds,
            nb_epochs=nb_epochs,
            seed=1,
            shuffle=shuffle,
            split_to_row_groups=True,
            shuffling_window=shuffling_window,
            files_circular_shift=files_circular_shift,
        ).and_return()

    spf = get_new_stream()

    all_fragments = list(iter(spf))
    all_fragments_bis = list(iter(get_new_stream()))

    assert set(list(Counter([f.stable_hash() for f in all_fragments]).values())) == set(
        [nb_epochs]
    )

    if shuffling_window <= total_number_of_row_groups // 2:
        assert set(
            list(
                Counter(
                    [f.stable_hash() for f in all_fragments[:shuffling_window]]
                ).values()
            )
        ) == set([1])

        assert set(
            list(Counter([f.stable_hash() for f in all_fragments]).values())
        ) == set([nb_epochs])

    # check deterministic behavior
    assert (
        len(all_fragments)
        == len(all_fragments_bis)
        == total_number_of_row_groups * nb_epochs
    )
    for f, f_bis in zip(all_fragments, all_fragments_bis):
        assert are_fragments_equal(f.fragment, f_bis.fragment)

    # reset
    spf.reset()
    all_fragments = list(iter(spf))
    assert (
        len(all_fragments)
        == len(all_fragments_bis)
        == total_number_of_row_groups * nb_epochs
    )
    for f, f_bis in zip(all_fragments, all_fragments_bis):
        assert are_fragments_equal(f.fragment, f_bis.fragment)

    # reload state
    spf.reset()
    ii = iter(spf)
    all_fragments = []
    for _ in range(min(stop_id, total_number_of_row_groups * nb_epochs)):
        all_fragments.append(next(ii))
    state = spf.state_dict(strict=False)
    state = pickle.loads(pickle.dumps(state))

    # starting new one
    spf = get_new_stream()
    spf.load_state_dict(state)
    all_fragments = all_fragments + list(iter(spf))
    assert (
        len(all_fragments)
        == len(all_fragments_bis)
        == total_number_of_row_groups * nb_epochs
    )
    for f, f_bis in zip(all_fragments, all_fragments_bis):
        assert are_fragments_equal(f.fragment, f_bis.fragment)


def test_stream_parquet_fragments_with_circular_shift(controled_row_groups_pq_dataset):
    parquet_ds = pq.ParquetDataset(controled_row_groups_pq_dataset)

    def show_fragments(frag):
        return (
            frag.fragment.path.split("/")[-2],
            [r.id for r in frag.fragment.row_groups][0],
        )

    def get_new_stream(files_circular_shift, shuffling_window):
        return (
            stream_parquet_fragments(
                parquet_ds,
                nb_epochs=2,
                seed=1,
                shuffle=True,
                split_to_row_groups=True,
                shuffling_window=shuffling_window,
                files_circular_shift=files_circular_shift,
            )
            .map(show_fragments)
            .and_return()
        )

    result_0 = list(iter(get_new_stream(0, 1)))
    # order of files (`cat`` here) is random and different from one epoch to another
    # order of row groups is the same
    expcted_0 = [
        ("cat=cat_0", 0),  # epoch 1
        ("cat=cat_0", 1),
        ("cat=cat_2", 0),
        ("cat=cat_2", 1),
        ("cat=cat_2", 2),
        ("cat=cat_1", 0),
        ("cat=cat_2", 0),  # epoch 2
        ("cat=cat_2", 1),
        ("cat=cat_2", 2),
        ("cat=cat_1", 0),
        ("cat=cat_0", 0),
        ("cat=cat_0", 1),
    ]
    assert result_0 == expcted_0

    result_05 = list(iter(get_new_stream(0.5, 1)))
    # files are shuffled by 1 = int(3 * 0.5) files left
    # 1 epoch (0, 2, 1) -> (2, 1, 0)
    # 2 epoch (2, 1, 0) -> (1, 0, 2)
    expcted_05 = [
        ("cat=cat_2", 0),  # epoch 1
        ("cat=cat_2", 1),
        ("cat=cat_2", 2),
        ("cat=cat_1", 0),
        ("cat=cat_0", 0),
        ("cat=cat_0", 1),
        ("cat=cat_1", 0),  # epoch 2
        ("cat=cat_0", 0),
        ("cat=cat_0", 1),
        ("cat=cat_2", 0),
        ("cat=cat_2", 1),
        ("cat=cat_2", 2),
    ]
    assert result_05 == expcted_05

    result_05_3 = list(iter(get_new_stream(0.5, 3)))
    # locally shuffled in window=3 elements from previous result
    expcted_05_3 = [
        ("cat=cat_2", 0),  # epoch 1
        ("cat=cat_2", 2),
        ("cat=cat_2", 1),
        ("cat=cat_0", 0),
        ("cat=cat_0", 1),
        ("cat=cat_1", 0),
        ("cat=cat_1", 0),  # epoch 2
        ("cat=cat_0", 1),
        ("cat=cat_0", 0),
        ("cat=cat_2", 0),
        ("cat=cat_2", 2),
        ("cat=cat_2", 1),
    ]
    assert result_05_3 == expcted_05_3


@pytest.mark.parametrize("nb_epochs", [1, 2, 10])
@pytest.mark.parametrize("shuffling_window", [-1, 0, 2, 4, 10])
def test_fragment_input_conifg(
    controled_row_groups_pq_dataset, nb_epochs, shuffling_window
):
    config = FragmentStreamingConfig(
        parquet_path=controled_row_groups_pq_dataset,
        nb_epochs=nb_epochs,
        seed=1,
        split_to_row_groups=True,
        fragment_shuffle_window=shuffling_window,
        files_circular_shift=True,
    )
    total_number_of_row_groups = 6
    # with files_circular_shift > 0, shards will be perfectly speparated

    PFS = ParquetFragmentStreamer(config=config)
    pi01 = PFS.build_pipeline(0, 1).map(lambda frag: frag.stable_hash()).and_return()
    pi02 = PFS.build_pipeline(0, 2).map(lambda frag: frag.stable_hash()).and_return()
    pi12 = PFS.build_pipeline(1, 2).map(lambda frag: frag.stable_hash()).and_return()

    result_01 = list(iter(pi01))
    result_02 = list(iter(pi02))
    result_12 = list(iter(pi12))

    if shuffling_window != -1:
        # always dijoint for files_circular_shift=True
        assert set(result_12) & set(result_02) == set()

    assert Counter(result_02) + Counter(result_12) == Counter(result_01)
    assert list(Counter(result_01).values()) == total_number_of_row_groups * [nb_epochs]
