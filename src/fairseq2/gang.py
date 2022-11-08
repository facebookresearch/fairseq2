# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from enum import Enum
from typing import final

import torch.distributed as dist
from overrides import final as finaloverride
from torch import Tensor
from torch.distributed import ProcessGroup, ReduceOp  # type: ignore[attr-defined]


class ReduceOperation(Enum):
    """Specifies a reduce operation."""

    SUM = 1
    """Sum"""
    MEAN = 2
    """Mean"""
    PRODUCT = 3
    """Product"""
    MIN = 4
    """Minimum"""
    MAX = 5
    """Maximum"""


class Gang(ABC):
    """Represents a set of processes that work collectively."""

    @abstractmethod
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        """Reduces the tensor across the gang.

        :param tensor:
            The input and output of the operation. The tensor will be modified
            in-place.
        :param op:
            The element-wise reduce operation.
        """


@final
class _ProcessGroupGang(Gang):
    def __init__(self, pg: ProcessGroup) -> None:
        self._pg = pg

    @finaloverride
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        dist.all_reduce(tensor, self._get_reduce_op(op), group=self._pg)

    @staticmethod
    def _get_reduce_op(op: ReduceOperation) -> ReduceOp:
        # FIXME, TODO: pybind type lookup is broken on ReduceOp in 1.13
        # this garbage is to have something that works 1.11 - 1.13

        o: ReduceOp

        if op == ReduceOperation.SUM:
            o = getattr(ReduceOp, "SUM")
        elif op == ReduceOperation.MEAN:
            o = getattr(ReduceOp, "AVG")
        elif op == ReduceOperation.PRODUCT:
            o = getattr(ReduceOp, "PRODUCT")
        elif op == ReduceOperation.MIN:
            o = getattr(ReduceOp, "MIN")
        elif op == ReduceOperation.MAX:
            o = getattr(ReduceOp, "MAX")
        else:
            raise ValueError(f"Unknown operation: {op}")

        # ReduceOp has badly broken type stubs at typing time;
        # this forces the type back to something sane.
        return o


def from_process_group(pg: ProcessGroup) -> Gang:
    """Wraps ``pg`` as a :class:`Gang`.

    :param pg:
        The process group to wrap.

    :returns:
        A :class:`Gang` instance that internally calls ``pg`` for collective
        operations.
    """
    return _ProcessGroupGang(pg)
