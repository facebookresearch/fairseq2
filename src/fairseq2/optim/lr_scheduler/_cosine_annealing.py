# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, final

from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.logging import log
from fairseq2.optim.lr_scheduler._handler import LRSchedulerHandler
from fairseq2.optim.lr_scheduler._lr_scheduler import (
    AbstractLRScheduler,
    LRScheduler,
    get_per_param_group,
)
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import ValidationError, ValidationResult, validate


@final
class CosineAnnealingLR(AbstractLRScheduler):
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

    _cycle_len: int
    _cycle_mul: float
    _num_warmup_steps: int
    _lr_mul: float
    _start_lrs: Sequence[float]
    _final_lrs: Sequence[float]

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
        :param optimizer:
            The optimizer to associate.
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
        self._cycle_len = cycle_len
        self._cycle_mul = cycle_mul
        self._num_warmup_steps = num_warmup_steps
        self._lr_mul = lr_mul

        self._start_lrs = get_per_param_group(optimizer, "start_lr", start_lr)
        self._final_lrs = get_per_param_group(optimizer, "final_lr", final_lr)

        super().__init__(optimizer, last_epoch)

    @override
    def _compute_lrs(self) -> list[float]:
        base_lrs = self.base_lrs

        # Linearly increase the learning rate to its base value during warmup.
        if self.last_epoch < self._num_warmup_steps:
            c = self.last_epoch / self._num_warmup_steps

            return [s + (b - s) * c for b, s in zip(base_lrs, self._start_lrs)]

        curr_step = self.last_epoch - self._num_warmup_steps

        # When each cycle has equal length, the computation is straightforward.
        if self._cycle_mul == 1.0:
            cycle_nr = curr_step // self._cycle_len

            cycle_len = self._cycle_len

            # The position of the step within the cycle.
            cycle_pos = curr_step - (cycle_nr * cycle_len)

        # Otherwise, it becomes a bit trickier. We have to treat the cycles as
        # a geometric series to find out the number, length, and offset of the
        # current cycle.
        else:
            mul = self._cycle_mul

            # Solve the equation \sum_{i=0}^{n} len(cycle_i) + x = step for n.
            cycle_nr = int(math.log(1 - curr_step / self._cycle_len * (1 - mul), mul))

            cycle_len = int(mul**cycle_nr * self._cycle_len)

            # Compute the sum of the lengths of the first `cycle_nr` cycles
            # (i.e. geometric series) which corresponds to the beginning offset
            # of the current cycle.
            cycle_offset = int((1 - mul**cycle_nr) / (1 - mul) * self._cycle_len)

            # The position of the step within the cycle.
            cycle_pos = curr_step - cycle_offset

        lr_mul = self._lr_mul**cycle_nr

        c = math.cos(math.pi * cycle_pos / cycle_len)

        min_lrs, max_lrs = self._final_lrs, base_lrs

        return [self._cycle_lr(mn, mx, lr_mul, c) for mn, mx in zip(min_lrs, max_lrs)]

    def _cycle_lr(self, min_lr: float, max_lr: float, lr_mul: float, c: float) -> float:
        min_lr *= lr_mul
        max_lr *= lr_mul

        return min_lr + 0.5 * (max_lr - min_lr) * (1 + c)


COSINE_ANNEALING_LR: Final = "cosine_annealing"


@dataclass(kw_only=True)
class CosineAnnealingLRConfig:
    cycle_len: int | None = None
    """The number of steps within the first cycle. If ``None``, will be set to
    ``num_steps - num_warmup_steps``."""

    num_warmup_steps: int = 0
    """The number of warmup steps."""

    cycle_mul: float = 1.0
    """The factor to grow the length of each cycle."""

    lr_mul: float = 1.0
    """The factor to scale the base and final learning rate at the end of each
    cycle."""

    start_lr: float = 0.0
    """The initial warmup learning rate."""

    final_lr: float | None = None
    """The final learning rate. If ``None``, :attr:`final_lr_scale` will be used."""

    final_lr_scale: float | None = 0.2
    """
    The optimizer learning rate will be scaled by this value to determine the
    final learning rate. If ``None``, :attr:`final_lr` will be used.
    """

    def validate(self) -> None:
        result = ValidationResult()

        if self.final_lr is not None:
            if self.final_lr_scale is not None:
                result.add_error(
                    "`final_lr` and `final_lr_scale` must not be specified at the same time."
                )
        elif self.final_lr_scale is None:
            result.add_error("Either `final_lr` or `final_lr_scale` must be specified.")

        if result.has_error:
            raise ValidationError(
                "The cosine-annealing learning rate scheduler configuration has one or more validation errors:", result  # fmt: skip
            )


@final
class CosineAnnealingLRHandler(LRSchedulerHandler):
    @override
    def create(
        self, optimizer: Optimizer, config: object, num_steps: int | None
    ) -> LRScheduler:
        config = structure(config, CosineAnnealingLRConfig)

        validate(config)

        if config.cycle_len is None:
            if num_steps is None:
                raise ValueError(
                    "`config.cycle_len` must be specified when `num_steps` is not specified."
                )

            cycle_len = num_steps - config.num_warmup_steps
        else:
            cycle_len = config.cycle_len

        if config.final_lr is not None and config.final_lr_scale is not None:
            raise ValueError(
                "`config.final_lr` and `config.final_lr_scale` must not be specified at the same time."
            )

        try:
            lr = optimizer.param_groups[0]["lr"]
        except (IndexError, KeyError):
            raise ValueError(
                "`optimizer` does not have a parameter group with an assigned learning rate."
            ) from None

        if config.final_lr_scale is not None:
            final_lr = lr * config.final_lr_scale
        elif config.final_lr is not None:
            final_lr = config.final_lr
        else:
            raise ValueError(
                "Either `config.final_lr` or `config.final_lr_scale` must be specified."
            )

        if final_lr > lr:
            log.warning("The final learning rate ({}) is greater than the optimizer learning rate ({}). This means the learning rate will increase over the course of the training.", final_lr, lr)  # fmt: skip

        return CosineAnnealingLR(
            optimizer,
            cycle_len,
            config.num_warmup_steps,
            cycle_mul=config.cycle_mul,
            lr_mul=config.lr_mul,
            start_lr=config.start_lr,
            final_lr=final_lr,
        )

    @property
    @override
    def requires_num_steps(self) -> bool:
        return False

    @property
    @override
    def config_kls(self) -> type[object]:
        return CosineAnnealingLRConfig
