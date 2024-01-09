# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairseq2.data import DataPipeline, DataPipelineError, read_sequence
from fairseq2.data.text import read_text
from fairseq2.typing import CPU
from tests.common import tmp_rng_seed


class TestSampleOp:
    def test_op_works(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7]).and_return()

        pipeline = DataPipeline.sample([pipeline1, pipeline2], [0.6, 0.4]).and_return()

        for _ in range(2):
            with tmp_rng_seed(CPU, seed=1234):
                assert list(pipeline) == [1, 2, 5, 3, 4, 1, 2, 3, 6, 7]

            pipeline.reset()

    def test_op_works_when_no_weight_is_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3]).and_return()
        pipeline2 = read_sequence([4, 5, 6]).and_return()
        pipeline3 = read_sequence([7, 8, 9]).and_return()

        pipeline = DataPipeline.sample([pipeline1, pipeline2, pipeline3]).and_return()

        for _ in range(2):
            with tmp_rng_seed(CPU, seed=1234):
                assert list(pipeline) == [4, 1, 5, 7, 2, 8, 9, 7, 8, 6, 3]

            pipeline.reset()

    def test_op_works_when_a_single_pipeline_is_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()

        pipeline = DataPipeline.sample([pipeline1]).and_return()

        for _ in range(2):
            with tmp_rng_seed(CPU, seed=1234):
                assert list(pipeline) == [1, 2, 3, 4]

            pipeline.reset()

    def test_op_works_when_no_pipeline_is_specified(self) -> None:
        pipeline = DataPipeline.sample([]).and_return()

        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iter(pipeline))

            pipeline.reset()

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

    def test_op_raises_error_when_weight_is_negative_or_zero(self) -> None:
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
        pipeline1 = read_text(pathname=" &^#").and_return()
        pipeline2 = read_text(pathname=" &^#").and_return()

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

        pipeline = DataPipeline.sample([pipeline1, pipeline2, pipeline3]).and_return()
        # [5, 1, 6, 0, 2, 2, 4, 6, 0, 7, 3, 2, 8, 4]

        d = None

        it = iter(pipeline)

        with tmp_rng_seed(CPU, seed=1234):
            # Move to the fifth example.
            for _ in range(5):
                d = next(it)

            assert d == 2

            rng = torch.get_rng_state()

            state_dict = pipeline.state_dict()

            # Read a few examples before we roll back.
            for _ in range(3):
                d = next(it)

            assert d == 6

            torch.set_rng_state(rng)

            # Expected to roll back to the fifth example.
            pipeline.load_state_dict(state_dict)

            # Move to EOD.
            for _ in range(9):
                d = next(it)

            assert d == 4

            rng = torch.get_rng_state()

            state_dict = pipeline.state_dict()

            pipeline.reset()

            torch.set_rng_state(rng)

            # Expected to be EOD.
            pipeline.load_state_dict(state_dict)

            with pytest.raises(StopIteration):
                next(iter(pipeline))
