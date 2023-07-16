# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import DataPipeline


class TestCountOp:
    def test_op_works(self) -> None:
        pipeline = DataPipeline.count(start=4).take(10).and_return()

        for _ in range(2):
            list(pipeline) == list(range(4, 15))

            pipeline.reset()

    def test_op_saves_and_restores_its_state(self) -> None:
        pipeline = DataPipeline.count(start=4).take(10).and_return()

        d = None

        it = iter(pipeline)

        # Move the the fifth example.
        for _ in range(5):
            d = next(it)

        assert d == 8

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(4):
            d = next(it)

        assert d == 12

        # Expected to roll back to the fifth example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(5):
            d = next(it)

        assert d == 13

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
