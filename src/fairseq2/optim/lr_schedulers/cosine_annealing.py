# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.optim.lr_schedulers.lr_scheduler import (
    AbstractLRScheduler,
    get_per_param_group,
)


@final
class CosineAnnealingLR(AbstractLRScheduler):
    """Represents the cosine annealing learning rate schedule described in
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

    Refer to the paper to learn more about the details.

    In addition to the original schedule, this implementation also supports a
    warmup phase where the learning rate is linearly increased for the first
    :math:`T_{warmup}` training steps to the base learning rate.

    .. note::

        This scheduler is not chainable.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        cycle_len: int,
        num_warmup_steps: int,
        *,
        cycle_mul: float = 1.0,
        lr_mul: float = 1.0,
        start_lr: float | Sequence[float] = 0.0,
        final_lr: float | Sequence[float] = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        :param optimizer: The optimizer to associate.
        :param cycle_len: The number of steps within the first cycle.
        :param num_warmup_steps: The number of warmup steps.
        :param cycle_mul: The factor to grow the length of each cycle.
        :param lr_mul: The factor to scale the base and final learning rate at
            the end of each cycle.
        :param start_lr: The initial warmup learning rate of all parameter
            groups or each group respectively.
        :param final_lr: The final learning rate of all parameter groups or
            each group respectively at the end of the first cycle.
        :param last_epoch: The index of the last epoch.
        """
        start_lrs = get_per_param_group(optimizer, "start_lr", start_lr)
        final_lrs = get_per_param_group(optimizer, "final_lr", final_lr)

        self.cycle_len = cycle_len
        self.cycle_mul = cycle_mul
        self.num_warmup_steps = num_warmup_steps
        self.lr_mul = lr_mul
        self.start_lrs = start_lrs
        self.final_lrs = final_lrs

        super().__init__(optimizer, last_epoch)

    @override
    def _compute_lrs(self) -> list[float]:
        base_lrs = self.base_lrs

        # Linearly increase the learning rate to its base value during warmup.
        if self.last_epoch < self.num_warmup_steps:
            c = self.last_epoch / self.num_warmup_steps

            return [s + (b - s) * c for b, s in zip(base_lrs, self.start_lrs)]

        curr_step = self.last_epoch - self.num_warmup_steps

        # When each cycle has equal length, the computation is straightforward.
        if self.cycle_mul == 1.0:
            cycle_nr = curr_step // self.cycle_len

            cycle_len = self.cycle_len

            # The position of the step within the cycle.
            cycle_pos = curr_step - (cycle_nr * cycle_len)

        # Otherwise, it becomes a bit trickier. We have to treat the cycles as
        # a geometric series to find out the number, length, and offset of the
        # current cycle.
        else:
            mul = self.cycle_mul

            # Solve the equation \sum_{i=0}^{n} len(cycle_i) + x = step for n.
            cycle_nr = int(math.log(1 - curr_step / self.cycle_len * (1 - mul), mul))

            cycle_len = int(mul**cycle_nr * self.cycle_len)

            # Compute the sum of the lengths of the first `cycle_nr` cycles
            # (i.e. geometric series) which corresponds to the beginning offset
            # of the current cycle.
            cycle_offset = int((1 - mul**cycle_nr) / (1 - mul) * self.cycle_len)

            # The position of the step within the cycle.
            cycle_pos = curr_step - cycle_offset

        lr_mul = self.lr_mul**cycle_nr

        c = math.cos(math.pi * cycle_pos / cycle_len)

        min_lrs, max_lrs = self.final_lrs, base_lrs

        def cycle_lr(min_lr: float, max_lr: float) -> float:
            min_lr *= lr_mul
            max_lr *= lr_mul

            return min_lr + 0.5 * (max_lr - min_lr) * (1 + c)

        return [cycle_lr(mn, mx) for mn, mx in zip(min_lrs, max_lrs)]
