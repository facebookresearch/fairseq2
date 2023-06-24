# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import read_sequence
from fairseq2.data.processors import StrToIntConverter


class TestMapOp:
    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 20])
    def test_op_works_as_expected(self, num_parallel_calls: int) -> None:
        def fn(d: int) -> int:
            return d**2

        dp = read_sequence([1, 2, 3, 4]).map(fn, num_parallel_calls).and_return()

        for _ in range(2):
            assert list(dp) == [1, 4, 9, 16]

            dp.reset()

    def test_op_works_with_data_processor_as_expected(self) -> None:
        fn = StrToIntConverter()

        dp = read_sequence(["1", "2", "3", "4"]).map(fn).and_return()

        for _ in range(2):
            assert list(dp) == [1, 2, 3, 4]

            dp.reset()

    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 20])
    def test_op_propagates_errors_as_expected(self, num_parallel_calls: int) -> None:
        def fn(d: int) -> int:
            if d == 3:
                raise ValueError("map error")

            return d

        dp = read_sequence([1, 2, 3, 4]).map(fn, num_parallel_calls).and_return()

        with pytest.raises(ValueError, match=r"^map error$"):
            for d in dp:
                pass

    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 20])
    def test_record_reload_position_works_as_expected(
        self, num_parallel_calls: int
    ) -> None:
        def fn(d: int) -> int:
            return d

        dp = read_sequence(list(range(1, 10))).map(fn, num_parallel_calls).and_return()

        d = None

        it = iter(dp)

        # Move the the second example.
        for _ in range(2):
            d = next(it)

        assert d == 2

        state_dict = dp.state_dict()

        # Read a few examples before we roll back.
        for _ in range(4):
            d = next(it)

        assert d == 6

        # Expected to roll back to the second example.
        dp.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(7):
            d = next(it)

        assert d == 9

        state_dict = dp.state_dict()

        dp.reset()

        # Expected to be EOD.
        dp.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(dp))
