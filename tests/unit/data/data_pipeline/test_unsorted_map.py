# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time
from itertools import islice

import pytest

from fairseq2.data import read_sequence


class TestUnsortedMapOp:
    @pytest.mark.parametrize("buffer_size,num_threads", [(0, 1), (1, 1), (4, 4)])
    def test_op_works(self, buffer_size: int, num_threads: int) -> None:
        def double_fn(d: int) -> int:
            return d * 2

        seq = list(range(1, 5))
        result_seq = list(range(2, 10, 2))

        pipeline = (
            read_sequence(seq)
            .unsorted_map(double_fn, buffer_size=buffer_size, num_threads=num_threads)
            .and_return()
        )

        for _ in range(2):
            assert set(pipeline) == set(result_seq)

            pipeline.reset()

    def test_op_yields_shortest_job_first(self) -> None:
        def sleep_fn(d: int) -> int:
            time.sleep(d / 100)
            return d

        seq = [3, 2, 1]
        result_seq = [1, 2, 3]

        pipeline = (
            read_sequence(seq)
            .unsorted_map(sleep_fn, buffer_size=3, num_threads=3)
            .and_return()
        )

        for _ in range(2):
            assert list(pipeline) == result_seq

            pipeline.reset()

    @pytest.mark.parametrize("buffer_size,num_threads", [(0, 1), (1, 1), (4, 4)])
    def test_op_works_after_reset(self, buffer_size: int, num_threads: int) -> None:
        def double_fn(d: int) -> int:
            time.sleep(d / 100)
            return d * 2

        seq = list(range(1, 10))
        result_seq = list(range(2, 20, 2))

        pipeline = (
            read_sequence(seq)
            .unsorted_map(double_fn, buffer_size=buffer_size, num_threads=num_threads)
            .and_return()
        )

        for _ in range(2):
            assert list(islice(pipeline, 5)) == result_seq[:5]

            pipeline.reset()

    @pytest.mark.parametrize("buffer_size,num_threads", [(0, 1), (1, 1), (4, 4)])
    def test_op_works_when_no_data_is_specified(
        self, buffer_size: int, num_threads: int
    ) -> None:
        pipeline = (
            read_sequence([])
            .unsorted_map(lambda x: x, buffer_size=buffer_size, num_threads=num_threads)
            .and_return()
        )

        for _ in range(2):
            assert list(pipeline) == []

            pipeline.reset()

    @pytest.mark.parametrize("buffer_size,num_threads", [(0, 1), (1, 1), (4, 4)])
    def test_op_propagates_errors(self, buffer_size: int, num_threads: int) -> None:
        def fn(d: int) -> int:
            if d == 6:
                raise ValueError("map error")

            return d

        seq = list(range(1, 10))

        pipeline = (
            read_sequence(seq)
            .unsorted_map(fn, buffer_size=buffer_size, num_threads=num_threads)
            .and_return()
        )

        with pytest.raises(ValueError) as exc_info:
            for d in pipeline:
                pass

        assert str(exc_info.value) == "map error"

    @pytest.mark.parametrize("buffer_size,num_threads", [(0, 1), (1, 1), (4, 4)])
    def test_op_saves_and_restores_its_state(
        self, buffer_size: int, num_threads: int
    ) -> None:
        def sleep_fn(d: int) -> int:
            time.sleep(d / 100)
            return d

        seq = list(range(1, 10))

        pipeline = (
            read_sequence(seq)
            .unsorted_map(sleep_fn, buffer_size=buffer_size, num_threads=num_threads)
            .and_return()
        )

        d = None

        it = iter(pipeline)

        # Move to the second example.
        for _ in range(2):
            d = next(it)

        assert d == 2

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(4):
            d = next(it)

        assert d == 6

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(7):
            d = next(it)

        assert d == 9

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
