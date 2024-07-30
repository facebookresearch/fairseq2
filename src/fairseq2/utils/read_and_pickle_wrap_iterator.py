from typing import Callable, Iterator, Tuple, TypeVar

from typing_extensions import Self

from fairseq2.data import DataPipelineBuilder, read_iterator

T = TypeVar("T")


class IteratorPickleWrapper(Iterator[T]):
    def __init__(self, iterator_fn: Callable[[], Iterator[T]]) -> None:
        self.iterator_fn = iterator_fn
        self.iterator: Iterator[T] = self.iterator_fn()
        self.counter = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        out = next(self.iterator)
        self.counter += 1
        return out

    def __getstate__(self) -> Tuple[Callable[[], Iterator[T]], int]:
        return self.iterator_fn, self.counter

    def __setstate__(self, state: Tuple[Callable[[], Iterator[T]], int]) -> None:
        self.iterator_fn, counter = state
        self.iterator = self.iterator_fn()
        for i in range(counter):
            next(self.iterator)
        self.counter = counter


def read_and_pickle_wrap_iterator(
    iterator_fn: Callable[[], Iterator[T]]
) -> DataPipelineBuilder:
    return read_iterator(
        IteratorPickleWrapper(iterator_fn),
        reset_fn=lambda x: IteratorPickleWrapper(iterator_fn),
        infinite=False,
    )
