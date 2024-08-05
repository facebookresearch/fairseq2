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

from fairseq2.optim.lr_scheduler.base import AbstractLRScheduler, _get_per_param_group


@final
class TriStageLR(AbstractLRScheduler):
    """Represents the tri-stage learning rate schedule as described in Section
    3.2 of :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.

     The learning rate schedule employs three stages:

       - The warm-up stage where the learning rate is linearly increased to its
         maximum value (i.e. `base_lr`)
       - The hold stage where the learning rate is kept constant at its maximum
         value.
       - The decay stage where the learning rate is exponentially decayed to its
         final value.

    .. note::
        This scheduler is not chainable.
    """

    _num_steps: int
    _start_lr_scales: Sequence[float]
    _final_lr_scales: Sequence[float]
    _start_lrs: Sequence[float] | None
    _final_lrs: Sequence[float] | None
    _num_stage1_steps: int
    _num_stage2_steps: int
    _num_stage3_steps: int

    def __init__(
        self,
        optimizer: Optimizer,
        num_steps: int,
        stage_ratio: tuple[float, float, float],
        *,
        start_lr_scale: float | Sequence[float] = 0.01,
        final_lr_scale: float | Sequence[float] = 0.01,
        last_epoch: int = -1,
    ) -> None:
        """
        :param optimizer:
            The optimizer to associate.
        :param num_steps:
            The total number of steps over which to adjust the learning rate.
        :param stage_ratio:
            The ratios of warmup, hold, and decay stages. Must add up to 1.
        :param start_lr_scale:
            The scale of the initial warm-up learning rate.
        :param final_lr_scale:
            The scale of the final learning rate.
        """
        if not math.isclose((s := sum(stage_ratio)), 1.0):
            raise ValueError(
                f"The sum of `stage_ratio` values must be 1.0, but is {s} instead."
            )

        self._num_steps = num_steps

        self._start_lr_scales = _get_per_param_group(
            optimizer, "start_lr", start_lr_scale
        )
        self._final_lr_scales = _get_per_param_group(
            optimizer, "final_lr", final_lr_scale
        )

        self._start_lrs = None
        self._final_lrs = None

        self._num_stage1_steps = int(stage_ratio[0] * num_steps)
        self._num_stage2_steps = int(stage_ratio[1] * num_steps)
        self._num_stage3_steps = int(stage_ratio[2] * num_steps)

        super().__init__(optimizer, last_epoch)

    @override
    def _compute_lrs(self) -> list[float]:
        base_lrs = self.base_lrs

        # Due to `LRScheduler`'s constructor quirks, we delay the initialization
        # of `start_lrs` and `final_lrs` to here.
        if self._start_lrs is None:
            self._start_lrs = [s * b for s, b in zip(self._start_lr_scales, base_lrs)]

        if self._final_lrs is None:
            self._final_lrs = [s * b for s, b in zip(self._final_lr_scales, base_lrs)]

        num_steps = self.last_epoch

        # Linearly increase the learning rate to its base value during warmup.
        if num_steps < self._num_stage1_steps:
            c = num_steps / self._num_stage1_steps

            return [s + (b - s) * c for b, s in zip(base_lrs, self._start_lrs)]

        num_steps -= self._num_stage1_steps

        # Keep the learning rate constant during second stage.
        if num_steps < self._num_stage2_steps:
            return list(base_lrs)

        num_steps -= self._num_stage2_steps

        if num_steps < self._num_stage3_steps:
            c = num_steps / self._num_stage3_steps

            return [b * math.exp(math.log(f) * c) for b, f in zip(base_lrs, self._final_lr_scales)]  # fmt: skip

        return list(self._final_lrs)
