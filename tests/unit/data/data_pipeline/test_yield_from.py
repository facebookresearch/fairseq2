# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

from fairseq2.data import DataPipeline, read_sequence


class TestYieldFromOp:
    def test_op_works(self) -> None:
        def fn(d: Tuple[int, int]) -> DataPipeline:
            a, b = d

            seq = list(range(a, b))

            return read_sequence(seq).and_return()

        pipeline = read_sequence([[1, 5], [9, 14]]).yield_from(fn).and_return()

        for _ in range(2):
            assert list(pipeline) == [1, 2, 3, 4, 9, 10, 11, 12, 13]

            pipeline.reset()

    def test_op_saves_and_restores_its_state(self) -> None:
        def fn(d: Tuple[int, int]) -> DataPipeline:
            a, b = d

            seq = list(range(a, b))

            return read_sequence(seq).and_return()

        pipeline = read_sequence([[1, 5], [9, 14]]).yield_from(fn).and_return()

        d = None

        it = iter(pipeline)

        # Move the the second example.
        for _ in range(2):
            d = next(it)

        assert d == 2

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(5):
            d = next(it)

        assert d == 11

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(7):
            d = next(it)

        assert d == 13

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
