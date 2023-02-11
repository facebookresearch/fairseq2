import logging
from typing import Iterable, Iterator, List, TypeVar

T = TypeVar("T")

log = logging.getLogger(__name__)


class RoundRobin(Iterable[T]):
    """RoundRobin cycles over the dataloader one by one.

    Yield items until the longest iterator is exhausted.
    Note that the iterator aren't reseted when an epoch starts, and each iterators
    continue from where they left.
    """

    def __init__(
        self,
        dataloaders: List[Iterable[T]],
    ):
        self.dataloaders = dataloaders

        self._iterators: List[Iterator[T]] = []
        self._epoch_done = [False for _ in self.dataloaders]

    def __iter__(self) -> Iterator[T]:
        if not self._iterators:
            self._iterators = [iter(d) for d in self.dataloaders]
        self._epoch_done = [False for _ in self.dataloaders]

        while True:
            for i, it in enumerate(self._iterators):
                try:
                    yield next(it)
                except StopIteration:
                    self._iterators[i] = iter(self.dataloaders[i])
                    self._epoch_done[i] = True
                    if sum(self._epoch_done) == len(self.dataloaders):
                        raise StopIteration
                    try:
                        yield next(self._iterators[i])
                    except StopIteration:
                        log.error(
                            f"Can't restart iterator {i}. Dataset {self.dataloaders[i]} looks empty"
                        )
                        raise StopIteration
