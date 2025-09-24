# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias, final

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing_extensions import override

LRScheduler: TypeAlias = _LRScheduler


class AbstractLRScheduler(ABC, LRScheduler):
    @final
    @override
    def get_lr(self) -> list[float]:  # type: ignore[override]
        if not self._get_lr_called_within_step:  # type: ignore[attr-defined]
            warnings.warn(
                "To get the last learning rate computed by the scheduler, use `get_last_lr()`."
            )

        return self._compute_lrs()

    @abstractmethod
    def _compute_lrs(self) -> list[float]:
        """Compute the learning rate of each parameter group."""


@final
class PassthroughLR(AbstractLRScheduler):
    def __init__(self, optimizer: Optimizer, *, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)

    @override
    def _compute_lrs(self) -> list[float]:
        return self.base_lrs


def get_per_param_group(
    optimizer: Optimizer, name: str, value: float | Sequence[float]
) -> Sequence[float]:
    num_param_groups = len(optimizer.param_groups)

    if isinstance(value, float):
        return [value] * num_param_groups

    if len(value) != num_param_groups:
        raise ValueError(
            f"The length of `{name}` must match the number of parameter groups ({num_param_groups}), but is {len(value)} instead."
        )

    return value


def get_effective_lr(scheduler: LRScheduler) -> float:
    """Return the effective learning rate computed by ``scheduler``."""
    return scheduler.get_last_lr()[0]
