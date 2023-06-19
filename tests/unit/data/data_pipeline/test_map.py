# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import read_sequence


class TestMapOp:
    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 20])
    def test_op_works_as_expected(self, num_parallel_calls: int) -> None:
        def fn(d: int) -> int:
            return d**2

        dp = read_sequence([1, 2, 3, 4]).map(fn, num_parallel_calls).and_return()

        for _ in range(2):
            output = []

            for d in dp:
                output.append(d)

            assert output == [1, 4, 9, 16]

            dp.reset()

    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 20])
    def test_op_propagates_errors_as_expected(self, num_parallel_calls: int) -> None:
        def fn(d: int) -> int:
            if d == 3:
                raise ValueError("The square of 3 cannot be taken.")

            return d**2

        dp = read_sequence([1, 2, 3, 4]).map(fn, num_parallel_calls).and_return()

        with pytest.raises(ValueError, match=r"^The square of 3 cannot be taken.$"):
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

        for _ in range(2):
            d = next(it)

        assert d == 2

        state_dict = dp.state_dict()

        for _ in range(4):
            d = next(it)

        assert d == 6

        dp.load_state_dict(state_dict)

        for _ in range(7):
            d = next(it)

        assert d == 9

        state_dict = dp.state_dict()

        dp.reset()

        dp.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(dp))
