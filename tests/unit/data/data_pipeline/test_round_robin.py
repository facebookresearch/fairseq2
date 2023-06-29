# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, NoReturn

import pytest

from fairseq2.data import DataPipeline, DataPipelineError, read_sequence


class TestZipOp:
    def test_op_works_as_expected(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()
        dp2 = read_sequence([5, 6, 7, 8]).and_return()
        dp3 = read_sequence([0, 2, 4, 6]).and_return()

        rdp = DataPipeline.round_robin([dp1, dp2, dp3]).and_return()

        for _ in range(2):
            assert list(rdp) == [1, 5, 0, 2, 6, 2, 3, 7, 4, 4, 8, 6]

            rdp.reset()

    def test_op_works_as_expected_with_single_data_pipeline(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()

        rdp = DataPipeline.round_robin([dp1]).and_return()

        for _ in range(2):
            assert list(rdp) == [1, 2, 3, 4]

            rdp.reset()

    def test_op_works_as_expected_with_empty_data_pipelines(self) -> None:
        dp1 = read_sequence([]).and_return()
        dp2 = read_sequence([]).and_return()
        dp3 = read_sequence([]).and_return()

        rdp = DataPipeline.round_robin([dp1, dp2, dp3]).and_return()

        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iter(rdp))

            rdp.reset()

    def test_op_works_as_expected_if_lengths_are_not_equal(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()
        dp2 = read_sequence([5, 6]).and_return()
        dp3 = read_sequence([]).and_return()
        dp4 = read_sequence([7, 8, 9, 0, 1, 2]).and_return()

        rdp = DataPipeline.round_robin([dp1, dp2, dp3, dp4]).and_return()

        for _ in range(2):
            assert list(rdp) == [1, 5, 7, 2, 6, 8, 3, 5, 9, 4, 6, 0, 1, 5, 1, 2, 6, 2]

            rdp.reset()

    def test_op_works_as_expected_with_no_data_pipeline(self) -> None:
        rdp = DataPipeline.round_robin([]).and_return()

        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iter(rdp))

            rdp.reset()

    def test_op_raises_error_if_one_of_the_pipelines_is_broken(self) -> None:
        def err(e: Any) -> NoReturn:
            raise ValueError()

        dp1 = read_sequence([1]).map(err).and_return()
        dp2 = read_sequence([1]).and_return()

        # Break the first pipeline.
        try:
            next(iter(dp1))
        except ValueError:
            pass

        with pytest.raises(
            DataPipelineError,
            match=r"^At least one of the specified data pipelines is broken and cannot be used in round robin\.$",
        ):
            DataPipeline.round_robin([dp1, dp2]).and_return()

    def test_record_reload_position_works_as_expected(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()
        dp2 = read_sequence([5, 6]).and_return()
        dp3 = read_sequence([]).and_return()
        dp4 = read_sequence([7, 8, 9, 0, 1, 2]).and_return()

        rdp = DataPipeline.round_robin([dp1, dp2, dp3, dp4]).and_return()

        d = None

        it = iter(rdp)

        # Move the the second example.
        for _ in range(5):
            d = next(it)

        assert d == 6

        state_dict = rdp.state_dict()

        # Read a few examples before we roll back.
        for _ in range(4):
            d = next(it)

        assert d == 9

        # Expected to roll back to the second example.
        rdp.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(13):
            d = next(it)

        assert d == 2

        state_dict = rdp.state_dict()

        rdp.reset()

        # Expected to be EOD.
        rdp.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(rdp))
