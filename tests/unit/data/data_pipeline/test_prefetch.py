# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import read_sequence


class TestPrefetchOp:
    @pytest.mark.parametrize("num_examples", [0, 1, 4, 20])
    def test_op_works_as_expected(self, num_examples: int) -> None:
        seq = list(range(1, 100))

        dp = read_sequence(seq).prefetch(num_examples).and_return()

        for _ in range(2):
            output = []

            for d in dp:
                output.append(d)

            assert output == seq

            dp.reset()

    @pytest.mark.parametrize("num_examples", [0, 1, 4, 20])
    def test_op_works_as_expected_if_reset(self, num_examples: int) -> None:
        seq = list(range(1, 100))

        dp = read_sequence(seq).prefetch(num_examples).and_return()

        for _ in range(2):
            it = iter(dp)

            output = []

            for c in range(50):
                output.append(next(it))

            assert output == seq[:50]

            dp.reset()

    @pytest.mark.parametrize("num_examples", [0, 1, 4, 20])
    def test_op_works_as_expected_with_no_data(self, num_examples: int) -> None:
        dp = read_sequence([]).prefetch(num_examples).and_return()

        for _ in range(2):
            output = []

            for d in dp:
                output.append(d)

            assert output == []

            dp.reset()

    @pytest.mark.parametrize("num_examples", [0, 1, 4, 20])
    def test_op_propagates_errors_as_expected(self, num_examples: int) -> None:
        def fn(d: int) -> int:
            if d == 60:
                raise ValueError("map error")

            return d

        seq = list(range(1, 100))

        dp = read_sequence(seq).map(fn).prefetch(num_examples).and_return()

        with pytest.raises(ValueError, match=r"^map error$"):
            for d in dp:
                pass

    @pytest.mark.parametrize("num_examples", [0, 1, 4, 20])
    def test_record_reload_position_works_as_expected(self, num_examples: int) -> None:
        seq = list(range(1, 100))

        dp = read_sequence(seq).prefetch(num_examples).and_return()

        d = None

        it = iter(dp)

        # Move the the second example.
        for _ in range(28):
            d = next(it)

        assert d == 28

        state_dict = dp.state_dict()

        # Read a few examples before we roll back.
        for _ in range(4):
            d = next(it)

        assert d == 32

        # Expected to roll back to the second example.
        dp.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(71):
            d = next(it)

        assert d == 99

        state_dict = dp.state_dict()

        dp.reset()

        # Expected to be EOD.
        dp.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(dp))
