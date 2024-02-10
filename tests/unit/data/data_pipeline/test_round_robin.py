# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import DataPipeline, DataPipelineError, read_sequence
from fairseq2.data.text import read_text


class TestRoundRobinOp:
    def test_op_works(self) -> None:
        pipeline1: DataPipeline[int] = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2: DataPipeline[int] = read_sequence([5, 6, 7, 8]).and_return()
        pipeline3: DataPipeline[int] = read_sequence([0, 2, 4, 6]).and_return()

        pipeline4: DataPipeline[int] = DataPipeline.round_robin(
            [pipeline1, pipeline2, pipeline3]
        ).and_return()

        for _ in range(2):
            assert list(pipeline4) == [1, 5, 0, 2, 6, 2, 3, 7, 4, 4, 8, 6]

            pipeline4.reset()

    def test_op_works_when_a_single_pipeline_is_specified(self) -> None:
        pipeline1: DataPipeline[int] = read_sequence([1, 2, 3, 4]).and_return()

        pipeline2: DataPipeline[int] = DataPipeline.round_robin(
            [pipeline1]
        ).and_return()

        for _ in range(2):
            assert list(pipeline2) == [1, 2, 3, 4]

            pipeline2.reset()

    def test_op_works_when_no_pipeline_is_specified(self) -> None:
        pipeline: DataPipeline[int] = DataPipeline.round_robin([]).and_return()

        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iter(pipeline))

            pipeline.reset()

    def test_op_works_when_infinite_pipeline_is_specified(self) -> None:
        pipeline1: DataPipeline[int] = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2: DataPipeline[int] = DataPipeline.constant(0).and_return()
        pipeline3: DataPipeline[int] = read_sequence([0, 2, 4, 6]).and_return()

        pipeline4: DataPipeline[int] = DataPipeline.round_robin(
            [pipeline1, pipeline2, pipeline3]
        ).and_return()

        for _ in range(2):
            assert list(pipeline4) == [1, 0, 0, 2, 0, 2, 3, 0, 4, 4, 0, 6]

            pipeline4.reset()

    def test_op_works_when_pipelines_are_empty(self) -> None:
        pipeline1: DataPipeline[int] = read_sequence([]).and_return()
        pipeline2: DataPipeline[int] = read_sequence([]).and_return()
        pipeline3: DataPipeline[int] = read_sequence([]).and_return()

        pipeline4: DataPipeline[int] = DataPipeline.round_robin(
            [pipeline1, pipeline2, pipeline3]
        ).and_return()

        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iter(pipeline4))

            pipeline4.reset()

    def test_op_works_when_pipelines_have_different_lengths(self) -> None:
        pipeline1: DataPipeline[int] = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2: DataPipeline[int] = read_sequence([5, 6]).and_return()
        pipeline3: DataPipeline[int] = read_sequence([]).and_return()
        pipeline4: DataPipeline[int] = read_sequence([7, 8, 9, 0, 1, 2]).and_return()

        pipeline5: DataPipeline[int] = DataPipeline.round_robin(
            [pipeline1, pipeline2, pipeline3, pipeline4]
        ).and_return()

        seq = [1, 5, 7, 2, 6, 8, 3, 5, 9, 4, 6, 0, 1, 5, 1, 2, 6, 2]

        for _ in range(2):
            assert list(pipeline5) == seq

            pipeline5.reset()

    def test_op_works_when_pipelines_stop_at_shortest_is_specified(self) -> None:
        pipeline1: DataPipeline[int] = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2: DataPipeline[int] = read_sequence([5, 6]).and_return()
        pipeline3: DataPipeline[int] = read_sequence([7, 8, 9, 0, 1, 2]).and_return()

        pipeline4: DataPipeline[int] = DataPipeline.round_robin(
            [pipeline1, pipeline2, pipeline3], stop_at_shortest=True
        ).and_return()

        seq = [1, 5, 7, 2, 6, 8]

        for _ in range(2):
            assert list(pipeline4) == seq

            pipeline4.reset()

    def test_op_raises_error_when_one_of_the_pipelines_is_broken(self) -> None:
        # Force a non-recoverable error.
        pipeline1: DataPipeline[int] = read_text(pathname=" &^#").and_return()
        pipeline2: DataPipeline[int] = read_text(pathname=" &^#").and_return()

        # Break the first pipeline.
        try:
            next(iter(pipeline1))
        except DataPipelineError:
            assert pipeline1.is_broken

        with pytest.raises(
            ValueError,
            match=r"^At least one of the specified data pipelines is broken and cannot be used in round robin\.$",
        ):
            DataPipeline.round_robin([pipeline1, pipeline2]).and_return()

    def test_op_saves_and_restores_its_state(self) -> None:
        pipeline1: DataPipeline[int] = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2: DataPipeline[int] = read_sequence([5, 6]).and_return()
        pipeline3: DataPipeline[int] = read_sequence([]).and_return()
        pipeline4: DataPipeline[int] = read_sequence([7, 8, 9, 0, 1, 2]).and_return()

        pipeline5: DataPipeline[int] = DataPipeline.round_robin(
            [pipeline1, pipeline2, pipeline3, pipeline4]
        ).and_return()

        d = None

        it = iter(pipeline5)

        # Move to the fifth example.
        for _ in range(5):
            d = next(it)

        assert d == 6

        state_dict = pipeline5.state_dict()

        # Read a few examples before we roll back.
        for _ in range(4):
            d = next(it)

        assert d == 9

        # Expected to roll back to the fifth example.
        pipeline5.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(13):
            d = next(it)

        assert d == 2

        state_dict = pipeline5.state_dict()

        pipeline5.reset()

        # Expected to be EOD.
        pipeline5.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline5))
