# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairseq2.data import DataPipeline, read_sequence
from fairseq2.data.text import read_text
from fairseq2.utils.version import is_pt2_or_greater
from tests.common import tmp_rng_seed

cpu_device = torch.device("cpu")


@pytest.mark.skipif(
    not is_pt2_or_greater(),
    reason="Different sampling results with versions lower than PyTorch 2.0",
)
class TestSampleOp:
    def test_op_works(self) -> None:
        dp1 = read_sequence([1, 2, 3]).and_return()
        dp2 = read_sequence([11, 12, 13]).and_return()
        rdp = DataPipeline.sample(
            [dp1, dp2], [0.5, 0.5], stop_at_shortest=True
        ).and_return()

        for _ in range(5):
            with tmp_rng_seed(cpu_device, seed=1234):
                assert list(rdp) == [11, 1, 12, 2, 3, 13]

                rdp.reset()

    def test_op_works_when_pipelines_have_different_lengths(self) -> None:
        dp1 = read_sequence([1, 2, 3]).and_return()
        dp2 = read_sequence([11, 12]).and_return()

        rdp = DataPipeline.sample(
            [dp1, dp2], [7, 4], stop_at_shortest=True
        ).and_return()

        for _ in range(2):
            with tmp_rng_seed(cpu_device, seed=1234):
                assert list(rdp) == [1, 2, 11, 3]

                rdp.reset()

    def test_op_works_when_no_weights_is_specified(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4, 5]).and_return()
        dp2 = read_sequence([11, 12]).and_return()
        dp3 = read_sequence([101, 102, 103]).and_return()
        rdp = DataPipeline.sample([dp1, dp2, dp3], stop_at_shortest=True).and_return()

        for _ in range(2):
            with tmp_rng_seed(cpu_device, seed=1234):
                assert list(rdp) == [11, 1, 12, 101, 2, 102, 103]

                rdp.reset()

    def test_op_works_when_weight_is_low(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4, 5]).and_return()
        dp2 = read_sequence([11, 12]).and_return()
        rdp = DataPipeline.sample(
            [dp1, dp2], [0.9, 0.1], stop_at_shortest=True
        ).and_return()

        for _ in range(2):
            with tmp_rng_seed(cpu_device, seed=1234):
                assert list(rdp) == [1, 2, 3, 4, 5]

                rdp.reset()

    def test_op_works_when_a_single_pipeline_is_specified(self) -> None:
        dp = read_sequence([1, 2, 3, 4, 5]).and_return()
        rdp = DataPipeline.sample([dp], stop_at_shortest=True).and_return()

        for _ in range(2):
            with tmp_rng_seed(cpu_device, seed=1234):
                assert list(rdp) == [1, 2, 3, 4, 5]

                rdp.reset()

    def test_op_works_when_pipelines_are_empty(self) -> None:
        dp1 = read_sequence([]).and_return()
        dp2 = read_sequence([]).and_return()
        dp3 = read_sequence([]).and_return()

        rdp = DataPipeline.sample([dp1, dp2, dp3]).and_return()

        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iter(rdp))

            rdp.reset()

    def test_op_works_when_seed_is_set_manually(self) -> None:
        dp1 = read_sequence([1, 2, 3]).and_return()
        dp2 = read_sequence([11, 12]).and_return()

        rdp = DataPipeline.sample(
            [dp1, dp2], [0.4, 0.6], stop_at_shortest=True
        ).and_return()

        for _ in range(2):
            with tmp_rng_seed(cpu_device, seed=1234):
                assert list(rdp) == [11, 1, 12, 2, 3]

                rdp.reset()

        for _ in range(2):
            with tmp_rng_seed(cpu_device, seed=5678):
                assert list(rdp) == [1, 11, 12]

                rdp.reset()

    def test_op_works_when_up_sampling(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4, 5]).and_return()
        dp2 = read_sequence([11, 12]).and_return()

        rdp = DataPipeline.sample(
            [dp1, dp2], [0.5, 0.5], stop_at_shortest=False
        ).and_return()
        for _ in range(2):
            with tmp_rng_seed(cpu_device, seed=1234):
                assert list(rdp) == [11, 1, 12, 2, 3, 11, 4, 12, 11, 12, 5]

                rdp.reset()

    def test_op_raises_error_when_weight_is_negative(self) -> None:
        dl1 = read_sequence([1, 2, 3, 4, 5]).and_return()
        dl2 = read_sequence([11, 12]).and_return()

        rdp = DataPipeline.sample([dl1, dl2], [0.5, -2]).and_return()

        with pytest.raises(
            RuntimeError,
            match=r"^probability tensor contains either `inf`, `nan` or element < 0",
        ):
            list(rdp)

    def test_op_raises_error_when_pipelines_and_weights_have_different_lengths(
        self,
    ) -> None:
        dl1 = read_sequence([1, 2, 3, 4, 5]).and_return()
        dl2 = read_sequence([11, 12]).and_return()

        with pytest.raises(
            ValueError,
            match=r"^The number of `pipelines` and the number of `weights` must be equal, but are 2 and 3 instead\.$",
        ):
            DataPipeline.sample([dl1, dl2], [0.3, 0.3, 2]).and_return()

    def test_op_raises_error_when_sum_of_weights_is_zero(self) -> None:
        dl1 = read_sequence([1, 2, 3, 4, 5]).and_return()
        dl2 = read_sequence([11, 12]).and_return()

        rdp = DataPipeline.sample([dl1, dl2], [0, 0]).and_return()

        with pytest.raises(
            RuntimeError,
            match=r"^invalid multinomial distribution \(sum of probabilities <= 0\)",
        ):
            list(rdp)

    def test_op_raises_error_when_no_pipeline_is_specified(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"^`pipelines` does not contain any elements\. Can not sample from empty set\.$",
        ):
            DataPipeline.sample([]).and_return()

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
        dp1 = read_sequence(list(range(7))).and_return()
        dp2 = read_sequence(list(range(10, 18))).and_return()
        dp3 = read_sequence(list(range(20, 26))).and_return()

        rdp = DataPipeline.sample([dp1, dp2, dp3], stop_at_shortest=True).and_return()
        # [10, 0, 11, 20, 1, 21, 22, 23, 24, 12, 2, 25, 13, 3]

        d = None

        it = iter(rdp)

        with tmp_rng_seed(cpu_device, seed=1234):
            # Move the the fifth example.
            for _ in range(5):
                d = next(it)

            assert d == 1

            state_dict = rdp.state_dict()

            gen_state = torch.get_rng_state()

            # Read a few examples before we roll back.
            for _ in range(7):
                d = next(it)

            assert d == 25

            # Expected to roll back to the fifth example.
            rdp.reset()

            rdp.load_state_dict(state_dict)

            torch.set_rng_state(gen_state)

            # Move to EOD.
            for _ in range(9):
                d = next(it)

            assert d == 3

            state_dict = rdp.state_dict()

            rdp.reset()

            # Expected to be EOD.
            rdp.load_state_dict(state_dict)

            with pytest.raises(StopIteration):
                next(iter(rdp))
