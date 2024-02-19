# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import DataPipeline, DataPipelineError, read_sequence
from fairseq2.data.text import read_text


class TestConcatOp:
    def test_op_works(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()

        pipeline = DataPipeline.concat([pipeline1, pipeline2]).and_return()

        for _ in range(2):
            assert list(pipeline) == [1, 2, 3, 4, 5, 6, 7, 8]

            pipeline.reset()

    def test_op_works_when_pipelines_are_empty(self) -> None:
        pipeline1 = read_sequence([]).and_return()
        pipeline2 = read_sequence([]).and_return()

        pipeline = DataPipeline.concat([pipeline1, pipeline2]).and_return()

        for _ in range(2):
            assert list(pipeline) == []

            pipeline.reset()

    def test_op_raises_error_when_one_of_the_pipelines_is_broken(self) -> None:
        # Force a non-recoverable error.
        pipeline1 = read_text(pathname=" &^#").and_return()
        pipeline2 = read_text(pathname=" &^#").and_return()

        # Break the first pipeline.
        try:
            next(iter(pipeline1))
        except DataPipelineError:
            assert pipeline1.is_broken

        with pytest.raises(
            ValueError,
            match=r"^At least one of the specified data pipelines is broken and cannot be concatenated\.$",
        ):
            DataPipeline.concat([pipeline1, pipeline2]).and_return()

    def test_op_saves_and_restores_its_state(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()

        pipeline = DataPipeline.concat([pipeline1, pipeline2]).and_return()

        d = None

        it = iter(pipeline)

        # Move to the second example.
        for _ in range(6):
            d = next(it)

        assert d == 6

        state_dict = pipeline.state_dict()

        # Read one more example before we roll back.
        d = next(it)

        assert d == 7

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(2):
            d = next(it)

        assert d == 8

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
