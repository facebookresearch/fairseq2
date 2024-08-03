# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import Union, final

from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.optim.lr_scheduler.base import AbstractLRScheduler, _get_per_param_group


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

        self._num_warmup_steps = num_warmup_steps

        self._start_lrs = _get_per_param_group(optimizer, "start_lr", start_lr)

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
