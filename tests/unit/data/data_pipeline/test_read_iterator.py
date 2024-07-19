# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import read_iterator

class TestDefaultIterator:
    def __init__(self):
        self.i = 0
    def reset(self):
        self.i = 0
    def __iter__(self):
        self.reset()
        return self
    def __next__(self):
        ret = self.i
        self.i += 1
        return ret

class TestEarlyStopIterator:
    def __init__(self, n):
        self.i = 0
        self.n = n
    def reset(self):
        self.i = 0
    def __iter__(self):
        self.reset()
        return self
    def __next__(self):
        ret = self.i
        if ret >= self.n:
            raise StopIteration
        self.i += 1
        return ret

class TestReadIteratorOp:
    def test_op_works(self) -> None:

        pipeline = read_iterator(TestDefaultIterator()).and_return()
        it = iter(pipeline)

        for _ in range(2):
            assert next(it) == 1
            assert next(it) == 2
            assert next(it) == 3

            pipeline.reset()

    def test_op_stops(self) -> None:

        pipeline = read_iterator(TestEarlyStopIterator(3)).and_return()
        it = iter(pipeline)

        for _ in range(2):
            assert next(it) == 1
            assert next(it) == 2
            with pytest.raises(
                StopIteration
            ):
                next(it)
            pipeline.reset()

    def test_op_works_when_input_iterator_is_empty(self) -> None:
        pipeline = read_iterator(TestEarlyStopIterator(0)).and_return()

        for _ in range(2):
            assert list(pipeline) == []

            pipeline.reset()

    def test_op_saves_and_restores_its_state(self) -> None:
        
        pipeline = read_iterator(TestIterator()).and_return()
        it = iter(pipeline)

        for _ in range(2):
            assert next(it) == 1

            state = pipeline.state_dict()

            assert next(it) == 2
            assert next(it) == 3

            pipeline.load_state_dict(state)

            assert next(it) == 2
            assert next(it) == 3
            assert next(it) == 4

            pipeline.reset()

    def test_op_saves_and_restores_its_state_with_finite_iterator(self) -> None:
        
        pipeline = read_iterator(TestEarlyStopIterator(3)).and_return()
        it = iter(pipeline)

        for _ in range(2):
            assert next(it) == 1

            state = pipeline.state_dict()

            assert next(it) == 2
            assert next(it) == 3

            pipeline.load_state_dict(state)

            assert next(it) == 2
            assert next(it) == 3
            assert next(it) == 4

            pipeline.reset()
