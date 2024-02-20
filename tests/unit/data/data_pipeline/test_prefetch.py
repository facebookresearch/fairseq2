# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import islice

import pytest

from fairseq2.data import DataPipelineError, read_sequence


class TestPrefetchOp:
    @pytest.mark.parametrize("num_examples", [0, 1, 4, 20])
    def test_op_works(self, num_examples: int) -> None:
        seq = list(range(1, 100))

        pipeline = read_sequence(seq).prefetch(num_examples).and_return()

        for _ in range(2):
            assert list(pipeline) == seq

            pipeline.reset()

    @pytest.mark.parametrize("num_examples", [0, 1, 4, 20])
    def test_op_works_after_reset(self, num_examples: int) -> None:
        seq = list(range(1, 100))

        pipeline = read_sequence(seq).prefetch(num_examples).and_return()

        for _ in range(2):
            assert list(islice(pipeline, 50)) == seq[:50]

            pipeline.reset()

    @pytest.mark.parametrize("num_examples", [0, 1, 4, 20])
    def test_op_works_when_no_data_is_specified(self, num_examples: int) -> None:
        pipeline = read_sequence([]).prefetch(num_examples).and_return()

        for _ in range(2):
            assert list(pipeline) == []

            pipeline.reset()

    @pytest.mark.parametrize("num_examples", [0, 1, 4, 20])
    def test_op_propagates_errors(self, num_examples: int) -> None:
        def fn(d: int) -> int:
            if d == 60:
                raise ValueError("map error")

            return d

        seq = list(range(1, 100))

        pipeline = read_sequence(seq).map(fn).prefetch(num_examples).and_return()

        with pytest.raises(DataPipelineError) as exc_info:
            for d in pipeline:
                pass

        cause = exc_info.value.__cause__

        assert isinstance(cause, ValueError)

        assert str(cause) == "map error"

    @pytest.mark.parametrize("num_examples", [0, 1, 4, 20])
    def test_op_saves_and_restores_its_state(self, num_examples: int) -> None:
        seq = list(range(1, 100))

        pipeline = read_sequence(seq).prefetch(num_examples).and_return()

        d = None

        it = iter(pipeline)

        # Move to the second example.
        for _ in range(28):
            d = next(it)

        assert d == 28

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(4):
            d = next(it)

        assert d == 32

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(71):
            d = next(it)

        assert d == 99

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
