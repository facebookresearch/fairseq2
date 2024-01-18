# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import read_sequence


class TestBucketOp:
    def test_op_works(self) -> None:
        seq = list(range(100))

        bucket_size = 4

        pipeline = read_sequence(seq).bucket(bucket_size).and_return()

        for _ in range(2):
            it = iter(pipeline)

            for i in range(25):
                d = next(it)

                offset = i * bucket_size

                assert d == [offset + i for i in range(4)]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_works_when_bucket_size_is_1(self) -> None:
        seq = list(range(100))

        pipeline = read_sequence(seq).bucket(1).and_return()

        for _ in range(2):
            it = iter(pipeline)

            for i in range(100):
                d = next(it)

                assert d == [i]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    @pytest.mark.parametrize("drop", [False, True])
    def test_op_works_when_final_bucket_is_partial(self, drop: bool) -> None:
        bucket_size = 7

        seq = list(range(100))

        pipeline = read_sequence(seq).bucket(bucket_size, drop).and_return()

        for _ in range(2):
            it = iter(pipeline)

            for i in range(14):
                d = next(it)

                offset = i * bucket_size

                assert d == [offset + i for i in range(7)]

            if not drop:
                d = next(it)

                assert d == [98, 99]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_raises_error_when_bucket_size_is_0(self) -> None:
        with pytest.raises(
            ValueError, match=r"^`bucket_size` must be greater than zero\.$"
        ):
            read_sequence(list(range(100))).bucket(0).and_return()

    def test_op_saves_and_restores_its_state(self) -> None:
        seq = list(range(1, 10))

        pipeline = read_sequence(seq).bucket(2).and_return()

        d = None

        it = iter(pipeline)

        # Move to the second example.
        for _ in range(2):
            d = next(it)

        assert d == [3, 4]

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(2):
            d = next(it)

        assert d == [7, 8]

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(3):
            d = next(it)

        assert d == [9]

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
