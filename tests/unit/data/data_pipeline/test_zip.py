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

        zdp = DataPipeline.zip([dp1, dp2, dp3]).and_return()

        for _ in range(2):
            assert list(zdp) == [[1, 5, 0], [2, 6, 2], [3, 7, 4], [4, 8, 6]]

            zdp.reset()

    def test_op_works_as_expected_with_single_data_pipeline(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()

        zdp = DataPipeline.zip([dp1]).and_return()

        for _ in range(2):
            assert list(zdp) == [[1], [2], [3], [4]]

            zdp.reset()

    def test_op_works_as_expected_with_no_data_pipeline(self) -> None:
        zdp = DataPipeline.zip([]).and_return()

        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iter(zdp))

            zdp.reset()

    def test_op_works_as_expected_with_names(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()
        dp2 = read_sequence([5, 6, 7, 8]).and_return()
        dp3 = read_sequence([0, 2, 4, 6]).and_return()

        zdp = DataPipeline.zip([dp1, dp2, dp3], names=["p1", "p2", "p3"]).and_return()

        for _ in range(2):
            assert list(zdp) == [
                {"p1": 1, "p2": 5, "p3": 0},
                {"p1": 2, "p2": 6, "p3": 2},
                {"p1": 3, "p2": 7, "p3": 4},
                {"p1": 4, "p2": 8, "p3": 6},
            ]

            zdp.reset()

    def test_op_raises_error_if_lengths_are_not_equal(self) -> None:
        dp1 = read_sequence([1, 2, 3]).and_return()
        dp2 = read_sequence([5, 6, 7, 8]).and_return()

        zdp = DataPipeline.zip([dp1, dp2]).and_return()

        with pytest.raises(
            DataPipelineError, match=r"^The zipped data pipelines are expected"
        ):
            for d in zdp:
                pass

    def test_op_warns_if_lengths_are_not_equal(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()
        dp2 = read_sequence([5, 6, 7]).and_return()
        dp3 = read_sequence([0, 2, 4, 6]).and_return()

        zdp = DataPipeline.zip([dp1, dp2, dp3], warn_only=True).and_return()

        for _ in range(2):
            assert list(zdp) == [[1, 5, 0], [2, 6, 2], [3, 7, 4]]

            # TODO: assert that warning is printed.

            zdp.reset()

    def test_op_raises_error_if_pipelines_and_names_do_not_match(self) -> None:
        dp1 = read_sequence([]).and_return()
        dp2 = read_sequence([]).and_return()

        with pytest.raises(
            ValueError,
            match=r"^The number of `pipelines` and the number of `names` must be equal, but are 2 and 3 instead\.$",
        ):
            DataPipeline.zip([dp1, dp2], ["p1", "p2", "p3"])

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
            match=r"^At least one of the specified data pipelines is broken and cannot be zipped\.$",
        ):
            DataPipeline.zip([dp1, dp2]).and_return()

    def test_record_reload_position_works_as_expected(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()
        dp2 = read_sequence([5, 6, 7, 8]).and_return()
        dp3 = read_sequence([0, 2, 4, 6]).and_return()

        zdp = DataPipeline.zip([dp1, dp2, dp3]).and_return()

        d = None

        it = iter(zdp)

        # Move the the second example.
        for _ in range(2):
            d = next(it)

        assert d == [2, 6, 2]

        state_dict = zdp.state_dict()

        # Read one more example before we roll back.
        d = next(it)

        assert d == [3, 7, 4]

        # Expected to roll back to the second example.
        zdp.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(2):
            d = next(it)

        assert d == [4, 8, 6]

        state_dict = zdp.state_dict()

        zdp.reset()

        # Expected to be EOD.
        zdp.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(zdp))
