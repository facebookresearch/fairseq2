# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import DataPipelineError, read_sequence, zip_data_pipelines


class TestZipDataPipelinesOp:
    def test_op_works_as_expected(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()
        dp2 = read_sequence([5, 6, 7, 8]).and_return()
        dp3 = read_sequence([0, 2, 4, 6]).and_return()

        zdp = zip_data_pipelines([dp1, dp2, dp3]).and_return()

        for _ in range(2):
            output = []

            for d in zdp:
                output.append(d)

            assert output == [[1, 5, 0], [2, 6, 2], [3, 7, 4], [4, 8, 6]]

            zdp.reset()

    def test_op_works_as_expected_with_single_data_pipeline(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()

        zdp = zip_data_pipelines([dp1]).and_return()

        for _ in range(2):
            output = []

            for d in zdp:
                output.append(d)

            assert output == [[1], [2], [3], [4]]

            zdp.reset()

    def test_op_works_as_expected_with_no_data_pipeline(self) -> None:
        zdp = zip_data_pipelines([]).and_return()

        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iter(zdp))

            zdp.reset()

    def test_op_raises_error_if_lengths_are_not_equal(self) -> None:
        dp1 = read_sequence([1, 2, 3]).and_return()
        dp2 = read_sequence([5, 6, 7, 8]).and_return()

        zdp = zip_data_pipelines([dp1, dp2]).and_return()

        with pytest.raises(
            DataPipelineError, match=r"^The zipped data pipelines are expected"
        ):
            for d in zdp:
                pass

    def test_op_warns_if_lengths_are_not_equal(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()
        dp2 = read_sequence([5, 6, 7]).and_return()
        dp3 = read_sequence([0, 2, 4, 6]).and_return()

        zdp = zip_data_pipelines([dp1, dp2, dp3], warn_only=True).and_return()

        for _ in range(2):
            output = []

            for d in zdp:
                output.append(d)

            assert output == [[1, 5, 0], [2, 6, 2], [3, 7, 4]]

            # TODO: assert that warning is printed.

            zdp.reset()

    def test_record_reload_position_works_as_expected(self) -> None:
        dp1 = read_sequence([1, 2, 3, 4]).and_return()
        dp2 = read_sequence([5, 6, 7, 8]).and_return()
        dp3 = read_sequence([0, 2, 4, 6]).and_return()

        zdp = zip_data_pipelines([dp1, dp2, dp3]).and_return()

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
