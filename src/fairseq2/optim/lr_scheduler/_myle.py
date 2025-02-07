# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, final

from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.optim.lr_scheduler._handler import LRSchedulerHandler
from fairseq2.optim.lr_scheduler._lr_scheduler import (
    AbstractLRScheduler,
    LRScheduler,
    get_per_param_group,
)
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import ValidationError, ValidationResult, validate


@final
class MyleLR(AbstractLRScheduler):
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

    _num_warmup_steps: int
    _start_lrs: Sequence[float]

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        *,
        start_lr: float | Sequence[float] = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        :param optimizer:
            The optimizer to associate.
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

        self._num_warmup_steps = num_warmup_steps

        self._start_lrs = get_per_param_group(optimizer, "start_lr", start_lr)

        super().__init__(optimizer, last_epoch)

    @override
    def _compute_lrs(self) -> list[float]:
        base_lrs = self.base_lrs

        # Linearly increase the learning rate to its base value during warmup.
        if self.last_epoch < self._num_warmup_steps:
            c = self.last_epoch / self._num_warmup_steps

            return [s + (b - s) * c for b, s in zip(base_lrs, self._start_lrs)]

        # After the warmup, decay the learning rate proportional to the inverse
        # square root of the step number.
        c = (self._num_warmup_steps / self.last_epoch) ** 0.5

        return [b * c for b in base_lrs]


MYLE_LR: Final = "myle"


@dataclass(kw_only=True)
class MyleLRConfig:
    num_warmup_steps: int = 1
    """The number of warmup steps."""

    start_lr: float = 0.0
    """The initial warmup learning rate."""

    def validate(self) -> None:
        result = ValidationResult()

        if self.num_warmup_steps == 0:
            result.add_error("`num_warmup_steps` must be greater than or equal to 1.")

        if result.has_error:
            raise ValidationError(
                "The Myle learning rate scheduler configuration has one or more validation errors:", result  # fmt: skip
            )


@final
class MyleLRHandler(LRSchedulerHandler):
    @override
    def create(
        self, optimizer: Optimizer, config: object, num_steps: int | None
    ) -> LRScheduler:
        config = structure(config, MyleLRConfig)

        validate(config)

        return MyleLR(optimizer, config.num_warmup_steps, start_lr=config.start_lr)

    @property
    @override
    def requires_num_steps(self) -> bool:
        return False

    @property
    @override
    def config_kls(self) -> type[object]:
        return MyleLRConfig
