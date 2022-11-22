import logging
from typing import TYPE_CHECKING, Iterable, Iterator, List

from fairseq2.generate.tokenizer import Tokenizer

if TYPE_CHECKING:
    from . import Batch

log = logging.getLogger(__name__)


class RoundRobin(Iterable["Batch"]):
    """RoundRobin cycles over the dataloader one by one.

    Yield items until the longest iterator is exhausted.
    Note that the iterator aren't reseted when an epoch starts, and each iterators
    continue from where they left.
    """

    def __init__(
        self,
        dataloaders: List[Iterable["Batch"]],
        *,
        tokenizer: Tokenizer,
        batch_first: bool,
    ):
        self.dataloaders = dataloaders
        self.tokenizer = tokenizer
        self.batch_first = batch_first

        self._iterators: List[Iterator["Batch"]] = []
        self._epoch_done = [False for _ in self.dataloaders]

    def __iter__(self) -> Iterator["Batch"]:
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
