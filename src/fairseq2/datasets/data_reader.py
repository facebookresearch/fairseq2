# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Any, Literal, TypeVar, final

from typing_extensions import Self, override

from fairseq2.data import DataPipeline
from fairseq2.datasets.utils import _min_num_batches, _sum_num_batches
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer

log = get_log_writer(__name__)


BatchT = TypeVar("BatchT")

BatchT_co = TypeVar("BatchT_co", covariant=True)


class DataReader(ABC, Iterator[list[BatchT_co]]):
    """Reads batches of examples from a dataset."""

    @abstractmethod
    def __iter__(self) -> Self:
        ...

    @abstractmethod
    def __next__(self) -> list[BatchT_co]:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state and move back to the first batch."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        ...

    @property
    @abstractmethod
    def num_accumulate(self) -> int:
        """The number of batches accumulated in each iteration."""


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
        sync_batches: bool = True,
        sync_mode: Literal["until_first", "until_last"] = "until_first",
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
        :param sync_mode:
            If ``until_first``, stops iteration when the first rank reaches end
            of data. If ``until_last``, stops iteration when the last rank
            reaches end of data; ranks that have already reached their end of
            data will return an empty list of batches.
        """
        self._pipeline = pipeline
        self._pipeline_iter = iter(pipeline)
        self._gang = gang
        self._num_accumulate = num_accumulate
        self._drop_remainder = drop_remainder
        self._sync_batches = sync_batches
        self._sync_until_last = sync_mode == "until_last"
        self._eod = False

    @override
    def __iter__(self) -> Self:
        return self

    @override
    def __next__(self) -> list[BatchT]:
        if self._eod:
            raise StopIteration()

        batches = []

        for idx in range(self._num_accumulate):
            try:
                batch = next(self._pipeline_iter)
            except StopIteration:
                break

            batches.append(batch)

        # If we read less than `num_accumulate` batches, it means we reached end
        # of data.
        if self._drop_remainder and len(batches) != self._num_accumulate:
            batches.clear()

        local_num_batches = len(batches)

        if self._sync_batches and self._gang.size > 1:
            if self._sync_until_last:
                num_batches = _sum_num_batches(local_num_batches, self._gang)
            else:
                num_batches = _min_num_batches(local_num_batches, self._gang, log)

                if num_batches != local_num_batches:
                    batches = batches[:num_batches]
        else:
            num_batches = local_num_batches

        self._eod = num_batches == 0

        if self._eod:
            raise StopIteration()

        return batches

    @override
    def reset(self) -> None:
        self._eod = False

        self._pipeline.reset()

    @override
    def state_dict(self) -> dict[str, Any]:
        return self._pipeline.state_dict()

    @override
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._eod = False

        self._pipeline.load_state_dict(state_dict)

    @property
    @override
    def num_accumulate(self) -> int:
        return self._num_accumulate
