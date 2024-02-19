# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import read_sequence


class TestReadSequenceOp:
    def test_op_works(self) -> None:
        seq = list(range(1, 10))

        pipeline = read_sequence(seq).and_return()

        for _ in range(2):
            assert list(pipeline) == seq

            pipeline.reset()

    def test_op_works_when_input_sequence_is_empty(self) -> None:
        pipeline = read_sequence([]).and_return()

        for _ in range(2):
            assert list(pipeline) == []

            pipeline.reset()

    def test_op_saves_and_restores_its_state(self) -> None:
        seq = list(range(1, 10))

        pipeline = read_sequence(seq).and_return()

        d = None

        it = iter(pipeline)

        # Move to the second example.
        for _ in range(2):
            d = next(it)

        assert d == 2

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(4):
            d = next(it)

        assert d == 6

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(7):
            d = next(it)

        assert d == 9

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
