# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.data.data_pipeline import read_sequence


class TestRepeatOp:
    def test_op_works(self) -> None:
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        pipeline = read_sequence(seq).repeat(3).and_return()

        output = seq * 3

        for _ in range(2):
            assert list(pipeline) == output

            pipeline.reset()

    def test_op_works_when_pipeline_is_empty(self) -> None:
        pipeline = read_sequence([]).repeat(3).and_return()

        for _ in range(2):
            assert list(pipeline) == []

            pipeline.reset()

    def test_op_works_when_num_repeats_is_zero(self) -> None:
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        pipeline = read_sequence(seq).repeat(0).and_return()

        for _ in range(2):
            assert list(pipeline) == []

            pipeline.reset()

    def test_op_saves_and_restores_its_state(self) -> None:
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        pipeline = read_sequence(seq).repeat(2).and_return()

        d = None

        it = iter(pipeline)

        # Move to the fifth example.
        for _ in range(13):
            d = next(it)

        assert d == 4

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(3):
            d = next(it)

        assert d == 7

        # Expected to roll back to the fifth example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(5):
            d = next(it)

        assert d == 9

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
