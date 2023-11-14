# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairseq2.data import Collater, read_sequence
from tests.common import assert_equal, device


class TestCollateOp:
    @pytest.mark.parametrize("pad_to_multiple", [1, 2, 3, 8])
    def test_op_works(self, pad_to_multiple: int) -> None:
        pad_value = 3

        bucket1 = [
            torch.full((4, 2), 0, device=device, dtype=torch.int64),
            torch.full((4, 2), 1, device=device, dtype=torch.int64),
            torch.full((4, 2), 2, device=device, dtype=torch.int64),
        ]

        bucket2 = [
            [{"foo1": 0, "foo2": 1}, {"foo3": 2, "foo4": 3}],
            [{"foo1": 4, "foo2": 5}, {"foo3": 6, "foo4": 7}],
            [{"foo1": 8, "foo2": 9}, {"foo3": 0, "foo4": 1}],
        ]

        seq = [bucket1, bucket2]

        pipeline = read_sequence(seq).collate(pad_value, pad_to_multiple).and_return()

        output1, output2 = list(pipeline)

        collater = Collater(pad_value, pad_to_multiple)

        expected_output1 = collater(bucket1)
        expected_output2 = collater(bucket2)

        assert_equal(output1["seqs"], expected_output1["seqs"])
        assert_equal(output1["seq_lens"], expected_output1["seq_lens"])

        assert output1["is_ragged"] == expected_output1["is_ragged"]

        assert output2 == expected_output2
