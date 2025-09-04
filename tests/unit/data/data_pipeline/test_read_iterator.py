# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from typing import TypeVar

import pytest
from typing_extensions import Self

from fairseq2.data.data_pipeline import DataPipelineError, read_iterator, read_sequence


class DefaultIterator(Iterator[int]):
    def __init__(self) -> None:
        self.i = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        ret = self.i
        self.i += 1
        return ret


class EarlyStopIterator(Iterator[int]):
    def __init__(self, n: int):
        self.i = 0
        self.n = n

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        ret = self.i
        if ret >= self.n:
            raise StopIteration
        self.i += 1
        return ret


T = TypeVar("T", DefaultIterator, EarlyStopIterator)


def reset_fn(iterator: T) -> T:
    iterator.i = 0
    return iterator


class TestReadIteratorOp:
    def test_op_works(self) -> None:
        pipeline = read_iterator(
            DefaultIterator(), reset_fn, infinite=True
        ).and_return()
        it = iter(pipeline)

        for _ in range(2):
            assert next(it) == 0
            assert next(it) == 1
            assert next(it) == 2

            pipeline.reset()

    def test_op_works_with_range(self) -> None:
        pipeline = read_iterator(
            iter(range(5)), lambda x: iter(range(5)), infinite=True
        ).and_return()
        it = iter(pipeline)

        for _ in range(2):
            assert next(it) == 0
            assert next(it) == 1
            assert next(it) == 2

            pipeline.reset()

        assert list(pipeline) == [*range(5)]

    def test_op_works_with_generator(self) -> None:
        def generator_fn() -> Iterator[int]:
            i = 0
            while True:
                yield i
                i += 1

        with pytest.raises(
            TypeError,
            match=r"^`iterator` is not pickleable; set `skip_pickling_check` to True"
            r" to bypass \(see `read_iterator` documentation for details\)\.$",
        ):
            pipeline = read_iterator(
                generator_fn(), lambda x: generator_fn(), infinite=True
            ).and_return()

        pipeline = read_iterator(
            generator_fn(),
            lambda x: generator_fn(),
            infinite=True,
            skip_pickling_check=True,
        ).and_return()

        it = iter(pipeline)

        for _ in range(2):
            assert next(it) == 0
            assert next(it) == 1
            assert next(it) == 2

            pipeline.reset()

        with pytest.raises(TypeError, match=r"^cannot pickle 'generator' object$"):
            pipeline.state_dict()

    def test_op_stops(self) -> None:
        pipeline = read_iterator(
            EarlyStopIterator(3), reset_fn, infinite=False
        ).and_return()
        it = iter(pipeline)

        for _ in range(2):
            assert next(it) == 0
            assert next(it) == 1
            assert next(it) == 2
            with pytest.raises(StopIteration):
                next(it)
            pipeline.reset()

    def test_op_works_when_input_iterator_is_empty(self) -> None:
        pipeline = read_iterator(
            EarlyStopIterator(0), reset_fn, infinite=False
        ).and_return()

        for _ in range(2):
            assert list(pipeline) == []

            pipeline.reset()

    def test_op_finitude_behavior(self) -> None:
        pipeline = (
            read_sequence([EarlyStopIterator(3)])
            .yield_from(
                lambda x: read_iterator(x, reset_fn, infinite=False).and_return()
            )
            .and_return()
        )

        for _ in range(2):
            assert list(pipeline) == [0, 1, 2]

            pipeline.reset()

        pipeline = (
            read_sequence([DefaultIterator()])
            .yield_from(
                lambda x: read_iterator(x, reset_fn, infinite=True).and_return()
            )
            .and_return()
        )

        with pytest.raises(
            DataPipelineError,
            match=r"^The data pipeline to yield from cannot be infinite\.$",
        ):
            next(iter(pipeline))

    def test_op_saves_and_restores_its_state(self) -> None:
        pipeline = read_iterator(
            DefaultIterator(), reset_fn, infinite=True
        ).and_return()
        it = iter(pipeline)

        for _ in range(2):
            assert next(it) == 0

            state = pipeline.state_dict()

            assert next(it) == 1
            assert next(it) == 2

            pipeline.load_state_dict(state)

            assert next(it) == 1
            assert next(it) == 2
            assert next(it) == 3

            pipeline.reset()

    def test_op_saves_and_restores_its_state_with_finite_iterator(self) -> None:
        pipeline = read_iterator(
            EarlyStopIterator(3), reset_fn, infinite=False
        ).and_return()
        it = iter(pipeline)

        for _ in range(2):
            assert next(it) == 0

            state = pipeline.state_dict()

            assert next(it) == 1
            assert next(it) == 2

            pipeline.load_state_dict(state)

            assert next(it) == 1
            assert next(it) == 2
            with pytest.raises(StopIteration):
                next(it)

            pipeline.reset()
