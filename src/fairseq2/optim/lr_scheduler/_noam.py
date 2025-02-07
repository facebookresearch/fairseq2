# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, final

from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.optim.lr_scheduler._handler import LRSchedulerHandler
from fairseq2.optim.lr_scheduler._lr_scheduler import AbstractLRScheduler, LRScheduler
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@final
class NoamLR(AbstractLRScheduler):
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

    _num_warmup_steps: int

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        *,
        last_epoch: int = -1,
    ) -> None:
        """
        :param optimizer:
            The optimizer to associate.
        :param num_warmup_steps:
            The number of warmup steps.
        :param last_epoch:
            The index of the last epoch.
        """
        self._num_warmup_steps = num_warmup_steps

        super().__init__(optimizer, last_epoch)

    @override
    def _compute_lrs(self) -> list[float]:
        # Linearly increase the learning rate during warmup.
        if self.last_epoch < self._num_warmup_steps:
            c = self.last_epoch * self._num_warmup_steps**-1.5

        # No warmup requested, decay from the base learning rate.
        elif self.last_epoch == 0:
            c = 1.0

        # After the warmup, decay the learning rate proportional to the inverse
        # square root of the step number.
        else:
            c = self.last_epoch**-0.5

        return [b * c for b in self.base_lrs]


NOAM_LR: Final = "noam"


@dataclass(kw_only=True)
class NoamLRConfig:
    num_warmup_steps: int = 0
    """The number of warmup steps."""


@final
class NoamLRHandler(LRSchedulerHandler):
    @override
    def create(
        self, optimizer: Optimizer, config: object, num_steps: int | None
    ) -> LRScheduler:
        config = structure(config, NoamLRConfig)

        validate(config)

        return NoamLR(optimizer, config.num_warmup_steps)

    @property
    @override
    def requires_num_steps(self) -> bool:
        return False

    @property
    @override
    def config_kls(self) -> type[object]:
        return NoamLRConfig
