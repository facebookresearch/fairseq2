# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import fairseq2.data.text


def test_round_robin_2dl() -> None:
    dl1 = fairseq2.data.read_sequence([1, 2, 3, 4, 5]).and_return()
    dl2 = fairseq2.data.read_sequence([11, 12]).and_return()
    r = fairseq2.data.round_robin_data_pipelines([dl1, dl2], []).and_return()

    expected = [1, 11, 2, 12, 3, 11, 4, 12, 5, 11, 1]
    assert list(r) == expected


def test_round_robin_3dl() -> None:
    dl1 = fairseq2.data.read_sequence([1, 2, 3, 4, 5]).and_return()
    dl2 = fairseq2.data.read_sequence([11, 12]).and_return()
    dl3 = fairseq2.data.read_sequence([101, 102, 103]).and_return()
    r = fairseq2.data.round_robin_data_pipelines([dl1, dl2, dl3], []).and_return()

    expected = [1, 11, 101, 2, 12, 102, 3, 11, 103, 4, 12, 101, 5, 11, 102, 1]

    assert list(r) == expected


def test_round_robin_3dl_skip() -> None:
    dl1 = fairseq2.data.read_sequence([1, 2, 3, 4, 5]).and_return()
    dl2 = fairseq2.data.read_sequence([11, 12]).and_return()
    dl3 = fairseq2.data.read_sequence([101, 102, 103]).and_return()
    r = fairseq2.data.round_robin_data_pipelines([dl1, dl2, dl3], []).and_return()

    expected = [1, 11, 101, 2, 12, 102, 3, 11, 103, 4, 12, 101, 5, 11, 102, 1]

    it = iter(r)
    nb_elements = 5
    assert r.skip(nb_elements) == nb_elements
    assert list(it) == expected[nb_elements:]


def test_round_robin_3dl_skip_with_offset() -> None:
    dl1 = fairseq2.data.read_sequence([1, 2, 3, 4, 5]).and_return()
    dl2 = fairseq2.data.read_sequence([11, 12]).and_return()
    dl3 = fairseq2.data.read_sequence([101, 102, 103]).and_return()
    r = fairseq2.data.round_robin_data_pipelines([dl1, dl2, dl3], []).and_return()

    expected = [1, 11, 101, 2, 12, 102, 3, 11, 103, 4, 12, 101, 5, 11, 102, 1]
    assert list(r) == expected

    it = iter(r)
    assert 1 == next(it)
    nb_elements = 5
    assert r.skip(nb_elements) == nb_elements
    actual = list(it)
    assert actual == expected[nb_elements + 1 :]


def test_round_robin_big_skip() -> None:
    dl1 = fairseq2.data.read_sequence([1, 2, 3, 4, 5]).and_return()
    dl2 = fairseq2.data.read_sequence([11, 12]).and_return()
    r = fairseq2.data.round_robin_data_pipelines([dl1, dl2], []).and_return()

    expected = [1, 11, 2, 12, 3, 11, 4, 12, 5, 11, 1]
    assert list(r) == expected

    it = iter(r)
    assert 1 == next(it)
    assert 11 == next(it)
    assert r.skip(5) == 5
    assert 12 == next(it)
    assert 5 == next(it)
    assert 4 == r.skip(4)  # we reset both datasources in that case
    with pytest.raises(StopIteration):
        next(it)
