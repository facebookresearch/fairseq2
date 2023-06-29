# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import read_sequence


class TestShardOp:
    def test_op_works_as_expected(self) -> None:
        seq = list(range(1, 23))

        dp = read_sequence(seq).shard(1, 5).and_return()

        for _ in range(2):
            assert list(dp) == [2, 7, 12, 17]

            dp.reset()

        dp = read_sequence(seq).shard(4, 5).and_return()

        for _ in range(2):
            assert list(dp) == [5, 10, 15, 20]

            dp.reset()

        seq = list(range(1, 4))

        dp = read_sequence(seq).shard(0, 5).and_return()

        for _ in range(2):
            assert list(dp) == []

            dp.reset()

    @pytest.mark.parametrize("idx", [4, 5])
    def test_op_raises_error_if_shard_idx_is_invalid(self, idx: int) -> None:
        with pytest.raises(
            ValueError,
            match=rf"^`shard_idx` must be less than `num_shards` \(4\), but is {idx} instead\.$",
        ):
            read_sequence([1, 2, 3, 4]).shard(idx, 4).and_return()

    def test_record_reload_position_works_as_expected(self) -> None:
        seq = list(range(1, 23))

        dp = read_sequence(seq).shard(2, 5).and_return()

        d = None

        it = iter(dp)

        # Move the the second example.
        for _ in range(2):
            d = next(it)

        assert d == 8

        state_dict = dp.state_dict()

        # Read one more example before we roll back.
        d = next(it)

        assert d == 13

        # Expected to roll back to the second example.
        dp.load_state_dict(state_dict)

        d = next(it)

        assert d == 13

        # Move to EOD.
        d = next(it)

        assert d == 18

        state_dict = dp.state_dict()

        dp.reset()

        # Expected to be EOD.
        dp.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(dp))
