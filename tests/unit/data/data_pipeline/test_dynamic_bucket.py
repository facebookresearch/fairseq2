# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import read_sequence


class TestDynamicBucketOp:
    def test_op_works(self) -> None:
        seq = list(range(1, 7))

        threshold = 6
        cost_fn = lambda x : x

        pipeline = read_sequence(seq).dynamic_bucket(threshold, cost_fn).and_return()

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == [1, 2, 3]
            assert next(it) == [4, 5]
            assert next(it) == [6]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_works_with_floats(self) -> None:
        seq = [0.1, 0.2, 0.3]

        threshold = 0.3
        cost_fn = lambda x : x

        pipeline = read_sequence(seq).dynamic_bucket(threshold, cost_fn).and_return()

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == [0.1, 0.2]
            assert next(it) == [0.3]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    @pytest.mark.parametrize("drop", [False, True])
    def test_op_works_when_final_bucket_is_partial(self, drop: bool) -> None:
        seq = [0, 1, 2, 3, 2]

        threshold = 3
        cost_fn = lambda x : x

        pipeline = read_sequence(seq).dynamic_bucket(threshold, cost_fn, drop_remainder=drop).and_return()

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == [0, 1, 2]
            assert next(it) == [3]

            if not drop:
                assert next(it) == [2]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_works_with_min_set(self) -> None:
        seq = list(range(1,11))

        threshold = 3
        cost_fn = lambda x : x

        pipeline = read_sequence(seq).dynamic_bucket(threshold, cost_fn, nb_min=2).and_return()

        for _ in range(2):
            it = iter(pipeline)

            d = None
            for i in range(1, 11, 2):
                d = next(it)
                assert d == [i, i + 1]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_works_with_max_set(self) -> None:
        seq = [0, 0, 0, 0, 1, 2, 3]

        threshold = 3
        cost_fn = lambda x : x

        pipeline = read_sequence(seq).dynamic_bucket(threshold, cost_fn, nb_max=2).and_return()

        for _ in range(2):
            it = iter(pipeline)
            
            assert next(it) == [0, 0]
            assert next(it) == [0, 0]
            assert next(it) == [1, 2]
            assert next(it) == [3]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_works_with_min_and_max_set(self) -> None:
        seq = [0, 0, 0, 0, 1, 2, 3, 4]

        threshold = 3
        cost_fn = lambda x : x

        pipeline = read_sequence(seq).dynamic_bucket(threshold, cost_fn, nb_min=2, nb_max=2).and_return()

        for _ in range(2):
            it = iter(pipeline)
            
            assert next(it) == [0, 0]
            assert next(it) == [0, 0]
            assert next(it) == [1, 2]
            assert next(it) == [3, 4]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    @pytest.mark.parametrize("drop", [False, True])
    def test_op_works_with_min_and_drop_set(self, drop: bool) -> None:
        seq = [1, 2, 4]

        threshold = 3
        cost_fn = lambda x : x

        pipeline = read_sequence(seq).dynamic_bucket(threshold, cost_fn, nb_min=2, drop_remainder=drop).and_return()

        for _ in range(2):
            it = iter(pipeline)
            
            assert next(it) == [1, 2]

            if not drop:
                assert next(it) == [4]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_raises_error_when_threshold_is_nonpositive(self) -> None:
        with pytest.raises(
            ValueError, match=r"^`threshold` must be greater than zero\.$"
        ):
            read_sequence(list(range(100))).dynamic_bucket(0, lambda x : 1).and_return()

        with pytest.raises(
            ValueError, match=r"^`threshold` must be greater than zero\.$"
        ):
            read_sequence(list(range(100))).dynamic_bucket(-1, lambda x : 1).and_return()

    def test_op_saves_and_restores_its_state(self) -> None:
        seq = list(range(1, 7))

        threshold = 2
        cost_fn = lambda x : x

        pipeline = read_sequence(seq).dynamic_bucket(threshold, cost_fn).and_return()

        d = None

        it = iter(pipeline)

        # Move to the second example.
        for _ in range(2):
            d = next(it)

        assert d == [3]

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(2):
            d = next(it)

        assert d == [5]

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(3):
            d = next(it)

        assert d == [6]

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
