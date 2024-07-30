from typing import Callable, Iterator, Tuple, TypeVar

from typing_extensions import Self

from fairseq2.data import DataPipelineBuilder, read_iterator

T = TypeVar("T")


class IteratorPickleWrapper(Iterator[T]):
    def __init__(self, iterator_factory: Callable[[], Iterator[T]]) -> None:
        self.iterator_factory = iterator_factory
        self.iterator: Iterator[T] = self.iterator_factory()
        self.counter = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        out = next(self.iterator)
        self.counter += 1
        return out

    def __getstate__(self) -> Tuple[Callable[[], Iterator[T]], int]:
        return self.iterator_factory, self.counter

    def __setstate__(self, state: Tuple[Callable[[], Iterator[T]], int]) -> None:
        self.iterator_factory, counter = state
        self.iterator = self.iterator_factory()
        for i in range(counter):
            next(self.iterator)
        self.counter = counter


def read_and_pickle_wrap_iterator(
    iterator_factory: Callable[[], Iterator[T]]
) -> DataPipelineBuilder:
    iterator = iterator_factory()
    try:
        return read_iterator(
            iterator, reset_fn=lambda x: iterator_factory(), infinite=False
        )
    except TypeError:
        return read_iterator(
            IteratorPickleWrapper(iterator_factory),
            reset_fn=lambda x: IteratorPickleWrapper(iterator_factory),
            infinite=False,
        )
