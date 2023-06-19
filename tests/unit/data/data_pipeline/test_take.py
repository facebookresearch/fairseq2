# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import read_sequence


class TestTakeOp:
    def test_op_works_as_expected(self) -> None:
        dp = read_sequence([1, 2, 3, 4, 5, 6, 7, 8, 9]).take(5).and_return()

        for _ in range(2):
            output = []

            for d in dp:
                output.append(d)

            assert output == [1, 2, 3, 4, 5]

            dp.reset()

    def test_op_works_if_count_is_larger_than_data(self) -> None:
        dp = read_sequence([1, 2, 3]).take(5).and_return()

        for _ in range(2):
            output = []

            for d in dp:
                output.append(d)

            assert output == [1, 2, 3]

            dp.reset()

    def test_op_works_if_count_is_zero(self) -> None:
        dp = read_sequence([1, 2, 3]).take(0).and_return()

        for _ in range(2):
            output = []

            for d in dp:
                output.append(d)

            assert output == []

            dp.reset()

    def test_record_reload_position_works_as_expected(self) -> None:
        dp = read_sequence([1, 2, 3, 4, 5, 6, 7, 8, 9]).take(5).and_return()

        d = None

        it = iter(dp)

        for _ in range(2):
            d = next(it)

        assert d == 2

        state_dict = dp.state_dict()

        for _ in range(2):
            d = next(it)

        assert d == 4

        dp.load_state_dict(state_dict)

        for _ in range(3):
            d = next(it)

        assert d == 5

        state_dict = dp.state_dict()

        dp.reset()

        dp.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(dp))
