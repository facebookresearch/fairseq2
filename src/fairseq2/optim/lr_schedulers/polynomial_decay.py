# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.optim.lr_schedulers.lr_scheduler import (
    AbstractLRScheduler,
    get_per_param_group,
)


@final
class PolynomialDecayLR(AbstractLRScheduler):
    """
    Represents the polynomial decay learning rate schedule.

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

    def __init__(
        self,
        optimizer: Optimizer,
        num_steps: int,
        num_warmup_steps: int,
        *,
        power: float = 1.0,
        start_lr: float | Sequence[float] = 0.0,
        final_lr: float | Sequence[float] = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        :param optimizer: The optimizer to associate.
        :param num_steps: The total number of steps, including warmup, over
            which to decay the learning rate.
        :param num_warmup_steps: The number of warmup steps.
        :param power: The exponent of the polynomial used for decay.
        :param start_lr: The initial warmup learning rate of all parameter
            groups or each group respectively.
        :param final_lr: The final learning rate of all parameter groups or each
            group respectively.
        :param last_epoch: The index of the last epoch.
        """
        if num_warmup_steps >= num_steps:
            raise ValueError(
                f"`num_warmup_steps` must be less than `num_steps` ({num_steps}), but is {num_warmup_steps} instead."
            )

        start_lrs = get_per_param_group(optimizer, "start_lr", start_lr)
        final_lrs = get_per_param_group(optimizer, "final_lr", final_lr)

        self.num_steps = num_steps
        self.num_warmup_steps = num_warmup_steps
        self.power = power
        self.start_lrs = start_lrs
        self.final_lrs = final_lrs

        super().__init__(optimizer, last_epoch)

    @override
    def _compute_lrs(self) -> list[float]:
        base_lrs = self.base_lrs

        # The decay is already complete, return the final learning rate.
        if self.last_epoch >= self.num_steps:
            return [f for f in self.final_lrs]

        # Linearly increase the learning rate to its base value during warmup.
        if self.last_epoch < self.num_warmup_steps:
            c = self.last_epoch / self.num_warmup_steps

            return [s + (b - s) * c for b, s in zip(base_lrs, self.start_lrs)]

        # After the warmup, decay the learning rate to its final value.
        r = self.num_steps - self.last_epoch
        t = self.num_steps - self.num_warmup_steps

        c = (r / t) ** self.power

        return [f + (b - f) * c for b, f in zip(base_lrs, self.final_lrs)]
