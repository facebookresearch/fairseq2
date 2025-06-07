# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.data.data_pipeline import read_sequence


class TestDynamicBucketOp:
    def test_op_works(self) -> None:
        seq = list(range(1, 7))

        threshold = 6
        cost_fn = lambda x: x

        pipeline = read_sequence(seq).dynamic_bucket(threshold, cost_fn).and_return()

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == [1, 2, 3]
            assert next(it) == [4, 5]
            assert next(it) == [6]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_works_with_square_cost_function(self) -> None:
        seq = list(range(1, 7))

        threshold = 14
        cost_fn = lambda x: x**2

        pipeline = read_sequence(seq).dynamic_bucket(threshold, cost_fn).and_return()

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == [1, 2, 3]
            assert next(it) == [4]
            assert next(it) == [5]
            assert next(it) == [6]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_works_with_length_cost_function(self) -> None:
        seq = [[1, 2], [3, 4, 5], [6], [7], [8, 9, 10], [11, 12, 13, 14, 15, 16]]

        threshold = 5
        cost_fn = lambda x: len(x)

        pipeline = read_sequence(seq).dynamic_bucket(threshold, cost_fn).and_return()

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == [[1, 2], [3, 4, 5]]
            assert next(it) == [[6], [7], [8, 9, 10]]
            assert next(it) == [[11, 12, 13, 14, 15, 16]]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_works_with_floats(self) -> None:
        seq = [0.1, 0.2, 0.3]

        threshold = 0.3
        cost_fn = lambda x: x

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
        cost_fn = lambda x: x

        pipeline = (
            read_sequence(seq)
            .dynamic_bucket(threshold, cost_fn, drop_remainder=drop)
            .and_return()
        )

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
        seq = list(range(1, 11))

        threshold = 3
        cost_fn = lambda x: x

        pipeline = (
            read_sequence(seq)
            .dynamic_bucket(threshold, cost_fn, min_num_examples=2)
            .and_return()
        )

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
        cost_fn = lambda x: x

        pipeline = (
            read_sequence(seq)
            .dynamic_bucket(threshold, cost_fn, max_num_examples=2)
            .and_return()
        )

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
        cost_fn = lambda x: x

        pipeline = (
            read_sequence(seq)
            .dynamic_bucket(threshold, cost_fn, min_num_examples=2, max_num_examples=2)
            .and_return()
        )

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
        cost_fn = lambda x: x

        pipeline = (
            read_sequence(seq)
            .dynamic_bucket(threshold, cost_fn, min_num_examples=2, drop_remainder=drop)
            .and_return()
        )

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == [1, 2]

            if not drop:
                assert next(it) == [4]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_works_with_bucket_creation_fn_set(self) -> None:
        seq = list(range(1, 7))

        threshold = 6
        cost_fn = lambda x: x
        bucket_creation_fn = lambda l: ([l[:-1]], [l[-1]])

        pipeline = (
            read_sequence(seq)
            .dynamic_bucket(
                threshold,
                cost_fn,
                bucket_creation_fn=bucket_creation_fn,
            )
            .and_return()
        )

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == [1, 2]
            assert next(it) == [3, 4]
            assert next(it) == [5]
            assert next(it) == [6]

            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()

    def test_op_raises_error_when_threshold_is_nonpositive(self) -> None:
        with pytest.raises(
            ValueError, match=r"^`threshold` must be greater than zero\.$"
        ):
            read_sequence(list(range(100))).dynamic_bucket(0, lambda x: 1).and_return()

    def test_op_raises_error_when_max_num_examples_is_nonpositive(self) -> None:
        with pytest.raises(
            ValueError, match=r"^`max_num_examples` must be greater than zero\.$"
        ):
            read_sequence(list(range(100))).dynamic_bucket(
                1, lambda x: 1, max_num_examples=0
            ).and_return()

    def test_op_raises_error_when_max_num_examples_is_less_than_min_num_examples(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"^`max_num_examples` must be greater than or equal to `min_num_examples`\.$",
        ):
            seq = read_sequence(list(range(100)))
            seq.dynamic_bucket(
                1, lambda x: 1, min_num_examples=2, max_num_examples=1
            ).and_return()

    def test_op_saves_and_restores_its_state(self) -> None:
        seq = list(range(1, 7))

        threshold = 2
        cost_fn = lambda x: x

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
