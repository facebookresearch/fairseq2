# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from typing import TypeVar, final

import torch
from typing_extensions import Self, override

from fairseq2.data.data_pipeline import DataPipeline, DataPipelineError
from fairseq2.gang import Gang, Gangs, all_sum
from fairseq2.logging import log
from fairseq2.typing import Stateful
from fairseq2.utils.tensor import to_tensor


class SyncMode(Enum):
    UNTIL_FIRST = 0
    """
    Stop data iteration on all ranks when one of the ranks reaches its end of
    data.
    """

    UNTIL_LAST = 1
    """
    Stop data iteration when all ranks reach their end of data; ranks that have
    already reached their end of data will return an empty list of batches.
    """


BatchT_co = TypeVar("BatchT_co", covariant=True)


class DataReader(ABC, Iterator[list[BatchT_co]], Stateful):
    """Reads batches of examples from a dataset."""

    @abstractmethod
    def __iter__(self) -> Self: ...

    @abstractmethod
    def __next__(self) -> list[BatchT_co]: ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state and move back to the first batch."""

    @abstractmethod
    def state_dict(self) -> dict[str, object]: ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, object]) -> None: ...

    @property
    @abstractmethod
    def num_accumulate(self) -> int:
        """The number of batches accumulated in each iteration."""


class DataReadError(Exception):
    pass


BatchT = TypeVar("BatchT")


@final
class DataPipelineReader(DataReader[BatchT]):
    """Reads batches of examples from a dataset using a :class:`DataPipeline`."""

    def __init__(
        self,
        pipeline: DataPipeline,
        gangs: Gangs,
        *,
        num_accumulate: int = 1,
        drop_remainder: bool = False,
        sync: bool = True,
        sync_mode: SyncMode = SyncMode.UNTIL_FIRST,
        strict_state: bool = True,
    ) -> None:
        """
        :param pipeline: The data pipeline to iterate over.
        :param gang: The gang over which the underlying dataset is sharded.
        :param strict_state: If ``True``, the entire state of the data pipeline
            including shuffling and bucketing buffers will be included in the
            state dictionary.
        """
        self._pipeline = pipeline
        self._pipeline_iter = iter(pipeline)
        self._gangs = gangs
        self._num_accumulate = num_accumulate
        self._drop_remainder = drop_remainder
        self._sync = sync
        self._sync_mode = sync_mode
        self._strict_state = strict_state
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
            except DataPipelineError as ex:
                raise DataReadError("Data pipeline failed to read next batch.") from ex

            batches.append(batch)

        # If we read less than `num_accumulate` batches, it means we reached end
        # of data.
        if self._drop_remainder and len(batches) != self._num_accumulate:
            batches.clear()

        local_num_batches = len(batches)

        if self._sync and self._gangs.dp.size > 1:
            if self._sync_mode == SyncMode.UNTIL_LAST:
                num_batches = _sum_num_batches(local_num_batches, self._gangs.dp)
            else:
                num_batches = _min_num_batches(local_num_batches, self._gangs.dp)

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
    def state_dict(self) -> dict[str, object]:
        return self._pipeline.state_dict(strict=self._strict_state)

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self._eod = False

        self._pipeline.load_state_dict(state_dict)

    @property
    @override
    def num_accumulate(self) -> int:
        return self._num_accumulate


def _min_num_batches(num_batches: int, gang: Gang) -> int:
    all_num_batches = torch.zeros((gang.size,), device=gang.device, dtype=torch.int64)

    num_batches_pt = to_tensor([num_batches], device=gang.device)

    gang.all_gather(all_num_batches, num_batches_pt)

    min_num_batches = int(all_num_batches.min())
    if min_num_batches != 0:
        return min_num_batches

    # If not all processes have reached end of data, report the ones that have
    # reached for debugging purposes.
    if log.is_enabled_for_debug():
        if all_num_batches.sum() > 0:
            ranks = all_num_batches.bool().logical_not_().nonzero().squeeze(-1).tolist()

            s = ", ".join(str(r) for r in ranks)

            log.debug("End of data reached at rank(s) {}.", s)

    return 0


def _sum_num_batches(num_batches: int, gang: Gang) -> int:
    total_num_batches = all_sum(gang, num_batches)

    return int(total_num_batches)
