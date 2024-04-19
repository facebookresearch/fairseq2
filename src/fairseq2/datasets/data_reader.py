# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Mapping, TypeVar, final

from typing_extensions import Self

from fairseq2.data import DataPipeline
from fairseq2.datasets.utils import _reduce_batch_stats
from fairseq2.gang import Gang
from fairseq2.models import Batch
from fairseq2.typing import override
from fairseq2.utils.logging import get_log_writer

log = get_log_writer(__name__)


BatchT = TypeVar("BatchT", bound=Batch)

BatchT_co = TypeVar("BatchT_co", bound=Batch, covariant=True)


@dataclass(frozen=True)
class DataOutput(Generic[BatchT]):
    """Represents the output of a :class:`DataReader`."""

    batches: List[BatchT]
    """The batches."""

    total_batch_size: int
    """The total size of batches read across all processes in the gang."""

    total_num_target_elements: int
    """The total number of target elements read across all processes in the gang."""


class DataReader(ABC, Iterator[DataOutput[BatchT_co]]):
    """Reads batches of examples from a dataset."""

    @abstractmethod
    def __iter__(self) -> Self:
        ...

    @abstractmethod
    def __next__(self) -> DataOutput[BatchT_co]:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state and move back to the first example."""

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        ...


@final
class DataPipelineReader(DataReader[BatchT]):
    """Reads batches of examples from a dataset using a :class:`DataPipeline`."""

    _pipeline: DataPipeline
    _pipeline_iter: Iterator[BatchT]
    _gang: Gang
    _num_accumulate: int
    _sync_batches: bool
    _eod: bool

    def __init__(
        self,
        pipeline: DataPipeline,
        gang: Gang,
        *,
        num_accumulate: int = 1,
        drop_remainder: bool = True,
        sync_batches: bool = False,
    ) -> None:
        """
        :param pipeline:
            The data pipeline to iterate over.
        :param gang:
            The gang over which the underlying dataset is sharded.
        :param num_accumulate:
            The number of batches to accumulate in each iteration. Typically
            used with gradient accumulation during training.
        :param drop_remainder:
            If ``True``, skips the last iteration if it returns less than
            ``num_accumulate`` batches.
        :param sync_batches:
            If ``True``, at the end of each ``next()`` call, syncs batches read
            across all processes in the gang. Typically used when the amount of
            data to be read can vary per process (e.g. due to bucketing) and it
            is critical for each process to iterate over same number of batches.
        """
        self._pipeline = pipeline
        self._pipeline_iter = iter(pipeline)
        self._gang = gang
        self._num_accumulate = num_accumulate
        self._drop_remainder = drop_remainder
        self._sync_batches = sync_batches
        self._eod = False

    @override
    def __iter__(self) -> Self:
        return self

    @override
    def __next__(self) -> DataOutput[BatchT]:
        if self._eod:
            raise StopIteration()

        batches = []

        stats = [(0, 0) for _ in range(self._num_accumulate)]

        for idx in range(self._num_accumulate):
            try:
                batch = next(self._pipeline_iter)
            except StopIteration:
                break

            batches.append(batch)

            stats[idx] = (batch.batch_size, batch.num_target_elements())

        # Gather batch stats.
        if self._sync_batches and self._gang.size > 1:
            num_batches, batch_size, num_target_elements = _reduce_batch_stats(
                stats, self._gang, log
            )

            batches = batches[:num_batches]
        else:
            # If we don't sync, we assume all processes read equal amount of
            # data at each iteration.
            batch_size = sum(s[0] for s in stats) * self._gang.size

            num_target_elements = sum(s[1] for s in stats) * self._gang.size

        # If we read less than `num_accumulate` batches, it means we reached end
        # of data.
        if self._drop_remainder and len(batches) != self._num_accumulate:
            batch_size = 0

        self._eod = batch_size == 0

        if self._eod:
            raise StopIteration()

        return DataOutput(batches, batch_size, num_target_elements)

    @override
    def reset(self) -> None:
        self._eod = False

        self._pipeline.reset()

    @override
    def state_dict(self) -> Dict[str, Any]:
        return self._pipeline.state_dict()

    @override
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._eod = False

        self._pipeline.load_state_dict(state_dict)
