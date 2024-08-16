from collections.abc import Callable, Iterator
from typing import TypeVar

from typing_extensions import Self, TypeAlias

from fairseq2.data import DataPipelineBuilder, read_iterator

T = TypeVar("T")

IteratorFactory: TypeAlias = Callable[[], Iterator[T]]


class IteratorPickleWrapper(Iterator[T]):
    def __init__(self, iterator_factory: IteratorFactory[T]) -> None:
        self._iterator_factory: IteratorFactory[T] = iterator_factory
        self._iterator: Iterator[T] = self._iterator_factory()
        self._counter = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        out = next(self._iterator)
        self._counter += 1
        return out

    def __getstate__(self) -> tuple[IteratorFactory[T], int]:
        return self._iterator_factory, self._counter

    def __setstate__(self, state: tuple[IteratorFactory[T], int]) -> None:
        self._iterator_factory, counter = state
        self._iterator = self._iterator_factory()
        for i in range(counter):
            next(self._iterator)
        self._counter = counter


def read_pickle_wrapped_iterator(
    iterator_factory: IteratorFactory[T],
) -> DataPipelineBuilder:
    iterator = iterator_factory()
    try:
        return read_iterator(
            iterator, reset_fn=lambda x: iterator_factory(), infinite=False
        )
    except TypeError as e:
        if (
            str(e)
            != "`iterator` is not pickleable; set `skip_pickling_check` to True to bypass (see `read_iterator` documentation for details)."
        ):
            raise
        return read_iterator(
            IteratorPickleWrapper(iterator_factory),
            reset_fn=lambda x: IteratorPickleWrapper(iterator_factory),
            infinite=False,
        )
