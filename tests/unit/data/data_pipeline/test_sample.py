# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from pathlib import Path

import pytest
import numpy as np

from fairseq2.data.data_pipeline import DataPipeline, DataPipelineError, read_sequence
from fairseq2.data.text import read_text


class TestSampleOp:
    def test_op_works(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7]).and_return()

        pipeline = DataPipeline.sample(
            [pipeline1, pipeline2], weights=[1.2, 0.8], seed=1234
        ).and_return()

        for _ in range(2):
            assert list(pipeline) == [1, 2, 3, 4, 1, 5, 2, 3, 6, 4, 7]

            pipeline.reset(reset_rng=True)

    def test_op_works_when_no_weight_is_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3]).and_return()
        pipeline2 = read_sequence([4, 5, 6]).and_return()
        pipeline3 = read_sequence([7, 8, 9]).and_return()

        pipeline = DataPipeline.sample(
            [pipeline1, pipeline2, pipeline3], seed=1234
        ).and_return()

        for _ in range(2):
            assert list(pipeline) == [1, 4, 2, 5, 3, 7, 1, 6, 8, 2, 9]

            pipeline.reset(reset_rng=True)

    def test_op_works_when_allow_repeats_false_is_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6]).and_return()
        pipeline3 = read_sequence([7, 8, 9, 10, 11, 12]).and_return()

        pipeline = DataPipeline.sample(
            [pipeline1, pipeline2, pipeline3],
            weights=[0.3, 0.6, 0.1],
            allow_repeats=False,
            seed=1234,
        ).and_return()

        for _ in range(2):
            assert list(pipeline) == [1, 5, 2, 6, 3, 4, 7, 8, 9, 10, 11, 12]

            pipeline.reset(reset_rng=True)

    def test_op_works_when_a_single_pipeline_is_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()

        pipeline = DataPipeline.sample([pipeline1], seed=1234).and_return()

        for _ in range(2):
            assert list(pipeline) == [1, 2, 3, 4]

            pipeline.reset(reset_rng=True)

    def test_op_works_when_no_pipeline_is_specified(self) -> None:
        pipeline = DataPipeline.sample([]).and_return()

        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iter(pipeline))

            pipeline.reset(reset_rng=True)

    def test_op_works_when_pseudo_infinite_pipeline_is_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = DataPipeline.count(5).and_return()

        pipeline = DataPipeline.sample(
            [pipeline1, pipeline2], weights=[0.4, 0.6], seed=1234
        ).and_return()

        for _ in range(2):
            assert list(pipeline) == [1, 5, 2, 3, 4]

            pipeline.reset(reset_rng=True)

    def test_op_works_when_infinite_pipeline_is_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).repeat().and_return()

        pipeline = DataPipeline.sample(
            [pipeline1, pipeline2], weights=[0.4, 0.6], seed=1234
        ).and_return()

        it = iter(pipeline)

        assert [next(it) for i in range(10)] == [1, 5, 2, 3, 4, 6, 1, 7, 8, 2]

    def test_op_raises_error_when_pipeline_is_empty(self) -> None:
        pipeline1 = read_sequence([1, 2]).and_return()
        pipeline2 = read_sequence([]).and_return()
        pipeline3 = read_sequence([3, 4]).and_return()

        pipeline = DataPipeline.sample([pipeline1, pipeline2, pipeline3]).and_return()

        with pytest.raises(
            DataPipelineError,
            match=r"^The data pipeline at index 1 is empty and cannot be sampled\.$",
        ):
            next(iter(pipeline))

    def test_op_raises_error_when_weight_is_not_valid(self) -> None:
        pipeline1 = read_sequence([1, 2]).and_return()
        pipeline2 = read_sequence([3, 4]).and_return()
        pipeline3 = read_sequence([5, 6]).and_return()

        with pytest.raises(
            ValueError,
            match=r"^The `weights` must be greater than 0\.0, but the weight at index 2 is -0\.2 instead\.$",
        ):
            DataPipeline.sample(
                [pipeline1, pipeline2, pipeline3], [0.5, 0.3, -0.2]
            ).and_return()

        with pytest.raises(
            ValueError,
            match=r"^The `weights` must be greater than 0\.0, but the weight at index 1 is 0 instead\.$",
        ):
            DataPipeline.sample(
                [pipeline1, pipeline2, pipeline3], [0.5, 0.0, 0.2]
            ).and_return()

        with pytest.raises(
            ValueError,
            match=r"^The `weights` must be finite, but the weight at index 0 is infinite or NaN instead\.$",
        ):
            DataPipeline.sample(
                [pipeline1, pipeline2, pipeline3], [math.inf, 0.3, 0.2]
            ).and_return()

    def test_op_raises_error_when_the_number_of_pipelines_and_weights_do_not_match(
        self,
    ) -> None:
        pipeline1 = read_sequence([1, 2, 3]).and_return()
        pipeline2 = read_sequence([4, 5, 6]).and_return()

        with pytest.raises(
            ValueError,
            match=r"^The number of `pipelines` and the number of `weights` must be equal, but are 2 and 3 instead\.$",
        ):
            DataPipeline.sample([pipeline1, pipeline2], [0.3, 0.3, 0.4]).and_return()

    def test_op_raises_error_when_one_of_the_pipelines_is_broken(self) -> None:
        # Force a non-recoverable error.
        pipeline1 = read_text(path=Path(" &^#")).and_return()
        pipeline2 = read_text(path=Path(" &^#")).and_return()

        # Break the first pipeline.
        try:
            next(iter(pipeline1))
        except Exception:
            assert pipeline1.is_broken

        with pytest.raises(
            ValueError,
            match=r"^At least one of the specified data pipelines is broken and cannot be sampled\.$",
        ):
            DataPipeline.sample([pipeline1, pipeline2]).and_return()

    def test_op_saves_and_restores_its_state(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()
        pipeline3 = read_sequence([0, 2, 4, 6]).and_return()

        # [1, 5, 2, 6, 3, 0, 4, 7, 2, 1, 4, 8, 6]
        pipeline = DataPipeline.sample(
            [pipeline1, pipeline2, pipeline3], seed=1234
        ).and_return()

        d = None

        it = iter(pipeline)

        # Move to the fifth example.
        for _ in range(5):
            d = next(it)

        assert d == 3

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(3):
            d = next(it)

        assert d == 7

        # Expected to roll back to the fifth example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(8):
            d = next(it)

        assert d == 6

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))

    def test_op_saves_and_restores_state_when_allow_repeats_false_is_specified(
        self,
    ) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6]).and_return()
        pipeline3 = read_sequence([7, 8, 9, 10, 11, 12]).and_return()

        pipeline = DataPipeline.sample(
            [pipeline1, pipeline2, pipeline3],
            weights=[0.3, 0.6, 0.1],
            allow_repeats=False,
            seed=1234,
        ).and_return()

        it = iter(pipeline)

        for _ in range(2):
            assert next(it) == 1
            assert next(it) == 5
            assert next(it) == 2
            assert next(it) == 6

            state_dict = pipeline.state_dict()

            assert list(pipeline) == [3, 4, 7, 8, 9, 10, 11, 12]

            pipeline.load_state_dict(state_dict)

            assert list(pipeline) == [3, 4, 7, 8, 9, 10, 11, 12]

            pipeline.reset(reset_rng=True)

    def test_op_works_with_many_pipelines_when_allow_repeats_false(self) -> None:
        """Test edge case with many pipelines where only lowest-weight ones remain."""
        
        # See https://github.com/fairinternal/fairseq2-ext/issues/181
        nb = 21
        weights = np.random.RandomState(0).rand(nb)
        sizes = np.random.RandomState(0).randint(0, 100_000, nb)
        pipelines = []
        for s in sizes:
            pipelines.append(read_sequence(list(range(s))).and_return())
        
        builder = DataPipeline.sample(pipelines, weights, seed=123, allow_repeats=False)
        pipeline = builder.and_return()

        expected_total_size = np.sum(sizes)
        actual_sampled_size = len([_ for _ in iter(pipeline)])
        assert actual_sampled_size == expected_total_size, f"Expected {expected_total_size}, got {actual_sampled_size}"
