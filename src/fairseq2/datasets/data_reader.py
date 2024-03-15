# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Mapping, TypeVar, final

from typing_extensions import Self

from fairseq2.data import DataPipeline
from fairseq2.datasets.utils import _all_eod
from fairseq2.gang import Gang
from fairseq2.typing import override

logger = logging.getLogger(__name__)


BatchT = TypeVar("BatchT")


class DataReader(ABC, Iterator[List[BatchT]]):
    """Reads batches of examples from a dataset."""

    @abstractmethod
    def __iter__(self) -> Self:
        ...

    @abstractmethod
    def __next__(self) -> List[BatchT]:
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
    _sync_eod: bool
    _eod: bool

    def __init__(
        self,
        pipeline: DataPipeline,
        gang: Gang,
        *,
        num_accumulate: int = 1,
        sync_eod: bool = False,
    ) -> None:
        """
        :param pipeline:
            The data pipeline to iterate over.
        :param gang:
            The gang across which the underlying dataset is sharded.
        :param num_accumulate:
            The number of batches to accumulate in each iteration. Typically
            used with gradient accumulation during training.
        :param sync_eod:
            If ``True``, syncs all processes in the gang about the end of data
            at the end of each ``next()`` call. Typically used when shards can
            have varying amount of data and it is critical for each process to
            iterate over the same number of batches (e.g. during training).
        """
        self._pipeline = pipeline
        self._pipeline_iter = iter(pipeline)
        self._gang = gang
        self._num_accumulate = num_accumulate
        self._sync_eod = sync_eod
        self._eod = False

    @override
    def __iter__(self) -> Self:
        return self

    @override
    def __next__(self) -> List[BatchT]:
        if self._eod:
            raise StopIteration()

        batches = []

        for _ in range(self._num_accumulate):
            try:
                batch = next(self._pipeline_iter)
            except StopIteration:
                break

            batches.append(batch)

        self._eod = len(batches) != self._num_accumulate

        # If requested, check the end of data across all processes in the gang.
        if self._sync_eod:
            self._eod = _all_eod(self._eod, self._gang, logger)

        if self._eod:
            raise StopIteration()

        return batches

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
