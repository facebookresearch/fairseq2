# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, NoReturn

import pytest

from fairseq2.data import DataPipeline, DataPipelineError, read_sequence


class TestZipOp:
    def test_op_works(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()
        pipeline3 = read_sequence([0, 2, 4, 6]).and_return()

        pipeline = DataPipeline.zip([pipeline1, pipeline2, pipeline3]).and_return()

        for _ in range(2):
            assert list(pipeline) == [[1, 5, 0], [2, 6, 2], [3, 7, 4], [4, 8, 6]]

            pipeline.reset()

    def test_works_when_a_single_pipeline_is_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()

        pipeline = DataPipeline.zip([pipeline1]).and_return()

        for _ in range(2):
            assert list(pipeline) == [[1], [2], [3], [4]]

            pipeline.reset()

    def test_op_works_when_no_pipeline_is_specified(self) -> None:
        pipeline = DataPipeline.zip([]).and_return()

        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iter(pipeline))

            pipeline.reset()

    def test_op_works_when_names_are_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()
        pipeline3 = read_sequence([0, 2, 4, 6]).and_return()

        pipeline = DataPipeline.zip(
            [pipeline1, pipeline2, pipeline3], names=["p1", "p2", "p3"]
        ).and_return()

        for _ in range(2):
            assert list(pipeline) == [
                {"p1": 1, "p2": 5, "p3": 0},
                {"p1": 2, "p2": 6, "p3": 2},
                {"p1": 3, "p2": 7, "p3": 4},
                {"p1": 4, "p2": 8, "p3": 6},
            ]

            pipeline.reset()

    def test_works_when_warn_only_is_true_and_pipelines_have_different_lengths(
        self,
    ) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7]).and_return()
        pipeline3 = read_sequence([0, 2, 4, 6]).and_return()

        pipeline = DataPipeline.zip(
            [pipeline1, pipeline2, pipeline3], warn_only=True
        ).and_return()

        for _ in range(2):
            assert list(pipeline) == [[1, 5, 0], [2, 6, 2], [3, 7, 4]]

            # TODO: assert that warning is printed.

            pipeline.reset()

    def test_raises_error_when_pipelines_have_different_lengths(self) -> None:
        pipeline1 = read_sequence([1, 2, 3]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()

        pipeline = DataPipeline.zip([pipeline1, pipeline2]).and_return()

        with pytest.raises(
            DataPipelineError, match=r"^The zipped data pipelines are expected"
        ):
            for d in pipeline:
                pass

    def test_raises_error_when_the_number_of_pipelines_and_names_do_not_match(
        self,
    ) -> None:
        pipeline1 = read_sequence([]).and_return()
        pipeline2 = read_sequence([]).and_return()

        with pytest.raises(
            ValueError,
            match=r"^The number of `pipelines` and the number of `names` must be equal, but are 2 and 3 instead\.$",
        ):
            DataPipeline.zip([pipeline1, pipeline2], ["p1", "p2", "p3"])

    def test_raises_error_when_one_of_the_pipelines_is_broken(self) -> None:
        def err(e: Any) -> NoReturn:
            raise ValueError()

        pipeline1 = read_sequence([1]).map(err).and_return()
        pipeline2 = read_sequence([1]).and_return()

        # Break the first pipeline.
        try:
            next(iter(pipeline1))
        except ValueError:
            pass

        with pytest.raises(
            DataPipelineError,
            match=r"^At least one of the specified data pipelines is broken and cannot be zipped\.$",
        ):
            DataPipeline.zip([pipeline1, pipeline2]).and_return()

    def test_saves_and_restores_its_state(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()
        pipeline3 = read_sequence([0, 2, 4, 6]).and_return()

        pipeline = DataPipeline.zip([pipeline1, pipeline2, pipeline3]).and_return()

        d = None

        it = iter(pipeline)

        # Move the the second example.
        for _ in range(2):
            d = next(it)

        assert d == [2, 6, 2]

        state_dict = pipeline.state_dict()

        # Read one more example before we roll back.
        d = next(it)

        assert d == [3, 7, 4]

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(2):
            d = next(it)

        assert d == [4, 8, 6]

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
