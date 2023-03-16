# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from abc import ABC, abstractmethod
from typing import List, Sequence, Union, final

from overrides import override as finaloverride
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing_extensions import TypeAlias

LRScheduler: TypeAlias = _LRScheduler


class LRSchedulerBase(ABC, LRScheduler):
    """Represents the abstract base class for learning rate schedulers."""

    @finaloverride
    def get_lr(self) -> List[float]:  # type: ignore[override]
        if not self._get_lr_called_within_step:  # type: ignore[attr-defined]
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`."
            )

        return self._compute_lrs()

    @abstractmethod
    def _compute_lrs(self) -> List[float]:
        """Compute the learning rate of each parameter group."""


@final
class NoamLR(LRSchedulerBase):
    """Represents the learning rate scheduler described in Section 5.3 of
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.

    .. math::
        lr = base\\_lr \\cdot \\min(\\frac{1}{\\sqrt{step\\_num}}, \\frac{step\\_num}{warmup\\_steps} \\cdot \\frac{1}{\\sqrt{warmup\\_steps}})

    This corresponds to increasing the learning rate linearly for the first
    *warmup_steps* training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number. In the paper, the authors use
    the square root of the dimensionality of the model as *base_lr*.

    This scheduler is commonly referred to as Noam, after the second author of
    the paper, Noam Shazeer.

    .. note::
        This scheduler is not chainable.
    """

    num_warmup_steps: int

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        :param optimizer:
            The associated optimizer.
        :param num_warmup_steps:
            The number of warmup steps.
        :param last_epoch:
            The index of the last epoch.
        :param verbose:
            If ``True``, prints a message to stdout for each update.
        """
        self.num_warmup_steps = num_warmup_steps

        super().__init__(optimizer, last_epoch, verbose)

    @finaloverride
    def _compute_lrs(self) -> List[float]:
        return [self._compute_lr(b) for b in self.base_lrs]

    def _compute_lr(self, base_lr: float) -> float:
        # Linearly increase the learning rate during warmup.
        if self.last_epoch < self.num_warmup_steps:
            return base_lr * self.last_epoch * self.num_warmup_steps**-1.5  # type: ignore[no-any-return]
        elif self.last_epoch == 0:
            # No warmup requested, decay from the base learning rate.
            return base_lr

        # After the warmup, decay the learning rate proportional to the inverse
        # square root of the step number.
        return base_lr * self.last_epoch**-0.5  # type: ignore[no-any-return]


@final
class MyleLR(LRSchedulerBase):
    """Represents a scaled version of :class:`NoamLR` that preserves the base
    learning rate of the associated optimizer.

    .. math::
        lr = base\\_lr \\cdot \\min(\\sqrt{\\frac{warmup\\_steps}{step\\_num}}, \\frac{step\\_num}{warmup\\_steps})

    Essentially, this is Noam learning rate schedule scaled by the square root
    of the number of warmup steps. It was originally proposed and implemented by
    Myle Ott in the original fairseq under the name ``InverseSquareRootLR``.

    It corresponds to increasing the learning rate linearly until the base
    learning rate, and decreasing it thereafter proportionally to the inverse
    square root of the step number.

    .. note::
        This scheduler is not chainable.
    """

    num_warmup_steps: int
    init_lrs: Sequence[float]

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        init_lr: Union[float, Sequence[float]] = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        :param optimizer:
            The associated optimizer.
        :param num_warmup_steps:
            The number of warmup steps.
        :param init_lr:
            The initial warmup learning rate of all parameter groups, or of each
            parameter group respectively.
        :param last_epoch:
            The index of the last epoch.
        :param verbose:
            If ``True``, prints a message to stdout for each update.
        """
        if num_warmup_steps == 0:
            raise ValueError("`num_warmup_steps` must be greater than 0.")

        self.num_warmup_steps = num_warmup_steps

        self.init_lrs = _get_per_param_group(optimizer, "init_lr", init_lr)

        super().__init__(optimizer, last_epoch, verbose)

    @finaloverride
    def _compute_lrs(self) -> List[float]:
        return [self._compute_lr(b, i) for b, i in zip(self.base_lrs, self.init_lrs)]

    def _compute_lr(self, base_lr: float, init_lr: float) -> float:
        # Linearly increase the learning rate to its base value during warmup.
        if self.last_epoch < self.num_warmup_steps:
            return (
                init_lr + self.last_epoch * (base_lr - init_lr) / self.num_warmup_steps
            )

        # After the warmup, decay the learning rate proportional to the inverse
        # square root of the step number.
        return base_lr * (self.num_warmup_steps / self.last_epoch) ** 0.5  # type: ignore[no-any-return]


def _get_per_param_group(
    optimizer: Optimizer, name: str, value: Union[float, Sequence[float]]
) -> Sequence[float]:
    num_param_groups = len(optimizer.param_groups)

    if isinstance(value, float):
        return [value] * num_param_groups

    if len(value) != num_param_groups:
        raise ValueError(
            f"The length of `{name}` ({len(value)}) does not match the number of parameter groups ({num_param_groups})."
        )

    return value
