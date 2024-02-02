# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from abc import ABC, abstractmethod
from typing import List, Sequence, Union, final

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing_extensions import TypeAlias

from fairseq2.typing import finaloverride

LRScheduler: TypeAlias = _LRScheduler


def get_effective_lr(scheduler: LRScheduler) -> float:
    """Return the effective learning rate computed by ``scheduler``."""
    return scheduler.get_last_lr()[0]


class LRSchedulerBase(ABC, LRScheduler):
    """Represents the abstract base class for learning rate schedulers."""

    @finaloverride
    def get_lr(self) -> List[float]:  # type: ignore[override]
        if not self._get_lr_called_within_step:  # type: ignore[attr-defined]
            warnings.warn(
                "To get the last learning rate computed by the scheduler, use `get_last_lr()`."
            )

        return self._compute_lrs()

    @abstractmethod
    def _compute_lrs(self) -> List[float]:
        """Compute the learning rate of each parameter group."""


@final
class NoamLR(LRSchedulerBase):
    """Represents the learning rate schedule described in Section 5.3 of
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.

    .. math::
        \\eta_t = \\eta_{base} \\min(\\frac{1}{\\sqrt{t}}, \\frac{t}{T_{warmup}} \\frac{1}{\\sqrt{T_{warmup}}})

    This corresponds to increasing the learning rate linearly for the first
    :math:`T_{warmup}` training steps, and decreasing it thereafter
    proportionally to the inverse square root of the step number. In the paper,
    the authors use the square root of the dimensionality of the model as
    :math:`\\eta_{base}`.

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
        *,
        last_epoch: int = -1,
    ) -> None:
        """
        :param optimizer:
            The associated optimizer.
        :param num_warmup_steps:
            The number of warmup steps.
        :param last_epoch:
            The index of the last epoch.
        """
        self.num_warmup_steps = num_warmup_steps

        super().__init__(optimizer, last_epoch)

    @finaloverride
    def _compute_lrs(self) -> List[float]:
        # Linearly increase the learning rate during warmup.
        if self.last_epoch < self.num_warmup_steps:
            c = self.last_epoch * self.num_warmup_steps**-1.5

        # No warmup requested, decay from the base learning rate.
        elif self.last_epoch == 0:
            c = 1.0

        # After the warmup, decay the learning rate proportional to the inverse
        # square root of the step number.
        else:
            c = self.last_epoch**-0.5

        return [b * c for b in self.base_lrs]


@final
class MyleLR(LRSchedulerBase):
    """Represents a scaled version of :class:`NoamLR` that preserves the base
    learning rate of the associated optimizer.

    .. math::
        \\eta_t = \\eta_{base} \\min(\\sqrt{\\frac{T_{warmup}}{t}}, \\frac{t}{T_{warmup}})

    Essentially, this is Noam learning rate schedule scaled by the square root
    of the number of warmup steps. It was originally proposed and implemented by
    Myle Ott in fairseq under the name ``InverseSquareRootLR``.

    It corresponds to increasing the learning rate linearly for the first
    :math:`T_{warmup}` training steps to the base learning rate, and decreasing
    it thereafter proportionally to the inverse square root of the step number.

    .. note::
        This scheduler is not chainable.
    """

    num_warmup_steps: int
    start_lrs: Sequence[float]

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        *,
        start_lr: Union[float, Sequence[float]] = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        :param optimizer:
            The associated optimizer.
        :param num_warmup_steps:
            The number of warmup steps.
        :param start_lr:
            The initial warmup learning rate of all parameter groups, or of each
            parameter group respectively.
        :param last_epoch:
            The index of the last epoch.
        """
        if num_warmup_steps == 0:
            raise ValueError("`num_warmup_steps` must be greater than 0.")

        self.num_warmup_steps = num_warmup_steps

        self.start_lrs = _get_per_param_group(optimizer, "start_lr", start_lr)

        super().__init__(optimizer, last_epoch)

    @finaloverride
    def _compute_lrs(self) -> List[float]:
        # Linearly increase the learning rate to its base value during warmup.
        if self.last_epoch < self.num_warmup_steps:
            c = self.last_epoch / self.num_warmup_steps

            return [s + (b - s) * c for b, s in zip(self.base_lrs, self.start_lrs)]

        # After the warmup, decay the learning rate proportional to the inverse
        # square root of the step number.
        c = (self.num_warmup_steps / self.last_epoch) ** 0.5

        return [b * c for b in self.base_lrs]


@final
class PolynomialDecayLR(LRSchedulerBase):
    """Represents the polynomial decay learning rate schedule.

    **During warmup:**

    .. math::
        \\eta_t = \\eta_{base} \\frac{t}{T_{warmup}}

    **After warmup:**

    .. math::
        \\eta_t = \\eta_{final} + (\\eta_{base} - \\eta_{final}) (\\frac{T - t}{T - T_{warmup}})^{p}

    This corresponds to increasing the learning rate linearly for the first
    :math:`T_{warmup}` training steps to the base learning rate, and decreasing
    it thereafter for :math:`T - T_{warmup}` steps to the final learning rate
    using a polynomial of degree :math:`p`.

    .. note::
        This scheduler is not chainable.
    """

    num_steps: int
    num_warmup_steps: int
    power: float
    start_lrs: Sequence[float]
    final_lrs: Sequence[float]

    def __init__(
        self,
        optimizer: Optimizer,
        num_steps: int,
        num_warmup_steps: int,
        *,
        power: float = 1.0,
        start_lr: Union[float, Sequence[float]] = 0.0,
        final_lr: Union[float, Sequence[float]] = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        :param optimizer:
            The associated optimizer.
        :param num_steps:
            The total number of steps, including warmup, over which to decay the
            learning rate.
        :param num_warmup_steps:
            The number of warmup steps.
        :param power:
            The exponent of the polynomial used for decay.
        :param start_lr:
            The initial warmup learning rate of all parameter groups, or of each
            parameter group respectively.
        :param final_lr:
            The final learning rate of all parameter groups, or of each
            parameter group respectively.
        :param last_epoch:
            The index of the last epoch.
        """
        if num_warmup_steps >= num_steps:
            raise ValueError(
                f"`num_warmup_steps` must be less than `num_steps` ({num_steps}), but is {num_warmup_steps} instead."
            )

        self.num_steps = num_steps
        self.num_warmup_steps = num_warmup_steps
        self.power = power

        self.start_lrs = _get_per_param_group(optimizer, "start_lr", start_lr)
        self.final_lrs = _get_per_param_group(optimizer, "final_lr", final_lr)

        super().__init__(optimizer, last_epoch)

    @finaloverride
    def _compute_lrs(self) -> List[float]:
        # The decay is already complete, return the final learning rate.
        if self.last_epoch >= self.num_steps:
            return [f for f in self.final_lrs]

        # Linearly increase the learning rate to its base value during warmup.
        if self.last_epoch <= self.num_warmup_steps:
            c = self.last_epoch / self.num_warmup_steps

            return [s + (b - s) * c for b, s in zip(self.base_lrs, self.start_lrs)]

        # After the warmup, decay the learning rate to its final value.
        r = self.num_steps - self.last_epoch
        t = self.num_steps - self.num_warmup_steps

        c = (r / t) ** self.power

        return [f + (b - f) * c for b, f in zip(self.base_lrs, self.final_lrs)]


@final
class CosineAnnealingLR(LRSchedulerBase):
    """Represents the learning rate schedule described in
    :cite:t:`https://doi.org/10.48550/arxiv.1608.03983`.

    **During warmup:**

    .. math::
        \\eta_t = \\eta_{base} \\frac{t}{T_{warmup}}

    **After warmup:**

    .. math::
        \\eta_t = \\eta_{final}^i + \\frac{1}{2} (\\eta_{base}^i - \\eta_{final}^i) (1 + \\text{cos}(\\pi \\frac{t_{i}}{T_{i}}))

    where :math:`i` is the number of the current annealing cycle, :math:`t_i` is
    the number of steps taken since the last restart, and :math:`T_i` is the
    total number of steps within the :math:`i`-th cycle (i.e. *length* of the
    cycle).

    *Cosine Annealing* is a type of learning rate schedule that has the effect
    of starting with a large learning rate that is relatively rapidly decreased
    to a minimum value before being increased rapidly again.

    Please refer to the paper to learn more about the details.

    In addition to the original schedule, this implementation also supports a
    warmup phase where the learning rate is linearly increased for the first
    :math:`T_{warmup}` training steps to the base learning rate.

    .. note::
        This scheduler is not chainable.
    """

    cycle_len: int
    cycle_mul: float
    num_warmup_steps: int
    lr_mul: float
    start_lrs: Sequence[float]
    final_lrs: Sequence[float]

    def __init__(
        self,
        optimizer: Optimizer,
        cycle_len: int,
        num_warmup_steps: int,
        *,
        cycle_mul: float = 1.0,
        lr_mul: float = 1.0,
        start_lr: Union[float, Sequence[float]] = 0.0,
        final_lr: Union[float, Sequence[float]] = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        :param optimizer:
            The associated optimizer.
        :param cycle_len:
            The number of steps within the first cycle.
        :param num_warmup_steps:
            The number of warmup steps.
        :param cycle_mul:
            The factor to grow the length of each cycle.
        :param lr_mul:
            The factor to scale the base and final learning rate at the end of
            each cycle.
        :param start_lr:
            The initial warmup learning rate of all parameter groups, or of each
            parameter group respectively.
        :param final_lr:
            The final learning rate of all parameter groups, or of each
            parameter group respectively, at the end of the first cycle.
        :param last_epoch:
            The index of the last epoch.
        """
        self.cycle_len = cycle_len
        self.cycle_mul = cycle_mul
        self.num_warmup_steps = num_warmup_steps
        self.lr_mul = lr_mul

        self.start_lrs = _get_per_param_group(optimizer, "start_lr", start_lr)
        self.final_lrs = _get_per_param_group(optimizer, "final_lr", final_lr)

        super().__init__(optimizer, last_epoch)

    @finaloverride
    def _compute_lrs(self) -> List[float]:
        # Linearly increase the learning rate to its base value during warmup.
        if self.last_epoch <= self.num_warmup_steps:
            c = self.last_epoch / self.num_warmup_steps

            return [i + (b - i) * c for b, i in zip(self.base_lrs, self.start_lrs)]

        curr_step = self.last_epoch - self.num_warmup_steps

        # When each cycle has equal length, the computation is straightforward.
        if self.cycle_mul == 1.0:
            cycle_num = curr_step // self.cycle_len

            cycle_len = self.cycle_len

            # The position of the step within the cycle.
            cycle_pos = curr_step - (cycle_num * cycle_len)

        # Otherwise, it becomes a bit trickier. We have to treat the cycles as
        # a geometric series to find out the number, length, and offset of the
        # current cycle.
        else:
            mul = self.cycle_mul

            # Solve the equation \sum_{i=0}^{n} len(cycle_i) + x = step for n.
            cycle_num = int(math.log(1 - curr_step / self.cycle_len * (1 - mul), mul))

            cycle_len = int(mul**cycle_num * self.cycle_len)

            # Compute the sum of the lengths of the first `cycle_num` cycles
            # (i.e. geometric series) which corresponds to the beginning offset
            # of the current cycle.
            cycle_offset = int((1 - mul**cycle_num) / (1 - mul) * self.cycle_len)

            # The position of the step within the cycle.
            cycle_pos = curr_step - cycle_offset

        lr_mul = self.lr_mul**cycle_num

        c = math.cos(math.pi * cycle_pos / cycle_len)

        min_lrs, max_lrs = self.final_lrs, self.base_lrs

        return [self._cycle_lr(mn, mx, lr_mul, c) for mn, mx in zip(min_lrs, max_lrs)]

    def _cycle_lr(self, min_lr: float, max_lr: float, lr_mul: float, c: float) -> float:
        min_lr *= lr_mul
        max_lr *= lr_mul

        return min_lr + 0.5 * (max_lr - min_lr) * (1 + c)


def _get_per_param_group(
    optimizer: Optimizer, name: str, value: Union[float, Sequence[float]]
) -> Sequence[float]:
    num_param_groups = len(optimizer.param_groups)

    if isinstance(value, float):
        return [value] * num_param_groups

    if len(value) != num_param_groups:
        raise ValueError(
            f"The length of `{name}` must be equal to the number of parameter groups ({num_param_groups}), but is {len(value)} instead."
        )

    return value
