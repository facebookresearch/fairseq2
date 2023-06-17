# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
