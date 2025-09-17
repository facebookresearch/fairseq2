# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch.optim import Optimizer

from fairseq2.error import InternalError
from fairseq2.logging import log
from fairseq2.optim.lr_schedulers import (
    CosineAnnealingLR,
    LRScheduler,
    MyleLR,
    NoamLR,
    PolynomialDecayLR,
    TriStageLR,
)
from fairseq2.recipe.component import ComponentManager, ComponentNotKnownError
from fairseq2.recipe.config import (
    COSINE_ANNEALING_LR,
    POLYNOMIAL_DECAY_LR,
    TRI_STAGE_LR,
    CosineAnnealingLRConfig,
    LRSchedulerSection,
    MyleLRConfig,
    NoamLRConfig,
    PolynomialDecayLRConfig,
    RegimeSection,
    TriStageLRConfig,
)
from fairseq2.recipe.error import LRSchedulerNotKnownError
from fairseq2.recipe.optim import maybe_raise_param_group_length_error
from fairseq2.utils.validation import ValidationError


@final
class _RecipeLRSchedulerFactory:
    def __init__(
        self, section: LRSchedulerSection, component_manager: ComponentManager
    ) -> None:
        self._section = section
        self._component_manager = component_manager

    def create(self) -> LRScheduler:
        section = self._section

        try:
            return self._component_manager.create_component(
                LRScheduler, section.name, section.config
            )
        except ComponentNotKnownError:
            raise LRSchedulerNotKnownError(section.name) from None


@final
class _CosineAnnealingLRFactory:
    def __init__(self, optimizer: Optimizer, regime_section: RegimeSection) -> None:
        self._optimizer = optimizer
        self._regime_section = regime_section

    def create(self, config: CosineAnnealingLRConfig) -> CosineAnnealingLR:
        if config.cycle_len is None:
            num_steps = self._regime_section.num_steps
            if num_steps is None:
                raise ValidationError(
                    f"`regime.num_steps` must be specified when `lr_scheduler` is '{COSINE_ANNEALING_LR}' and `lr_scheduler.config.cycle_len` is not specified."
                )

            cycle_len = num_steps - config.num_warmup_steps
        else:
            cycle_len = config.cycle_len

        optimizer = self._optimizer

        lrs = []

        for idx, param_group in enumerate(optimizer.param_groups):
            try:
                lr: float = param_group["lr"]
            except KeyError:
                raise InternalError(
                    f"`optimizer.param_groups[{idx}]` does not have a learning rate."
                ) from None

            lrs.append(lr)

        num_param_groups = len(optimizer.param_groups)

        start_lrs = config.start_lr

        if isinstance(start_lrs, float):
            start_lrs = [start_lrs] * num_param_groups
        else:
            maybe_raise_param_group_length_error(
                "start_lr", start_lrs, num_param_groups
            )

        if config.final_lr is not None:
            if config.final_lr_scale is not None:
                raise InternalError(
                    "`config.final_lr` and `config.final_lr_scale` are both specified."
                )

            final_lrs = config.final_lr

            if isinstance(final_lrs, float):
                final_lrs = [final_lrs] * num_param_groups
            else:
                maybe_raise_param_group_length_error(
                    "final_lr", final_lrs, num_param_groups
                )
        else:
            final_lr_scales = config.final_lr_scale
            if final_lr_scales is None:
                raise InternalError(
                    "`config.final_lr` and `config.final_lr_scale` are both `None`."
                )

            if isinstance(final_lr_scales, float):
                final_lr_scales = [final_lr_scales] * num_param_groups
            else:
                maybe_raise_param_group_length_error(
                    "final_lr_scale", final_lr_scales, num_param_groups
                )

            final_lrs = [lr * scale for lr, scale in zip(lrs, final_lr_scales)]

        for idx, (lr, final_lr) in enumerate(zip(lrs, final_lrs)):
            if final_lr > lr:
                log.warning("The final learning rate ({}) of optimizer parameter group {} is greater than the learning rate ({}). This means the learning rate will increase over the course of the training.", final_lr, idx, lr)  # fmt: skip

        return CosineAnnealingLR(
            optimizer,
            cycle_len,
            config.num_warmup_steps,
            cycle_mul=config.cycle_mul,
            lr_mul=config.lr_mul,
            start_lr=start_lrs,
            final_lr=final_lrs,
        )


@final
class _MyleLRFactory:
    def __init__(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    def create(self, config: MyleLRConfig) -> MyleLR:
        num_param_groups = len(self._optimizer.param_groups)

        start_lrs = config.start_lr

        if isinstance(start_lrs, float):
            start_lrs = [start_lrs] * num_param_groups
        else:
            maybe_raise_param_group_length_error(
                "start_lr", start_lrs, num_param_groups
            )

        return MyleLR(self._optimizer, config.num_warmup_steps, start_lr=start_lrs)


@final
class _NoamLRFactory:
    def __init__(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    def create(self, config: NoamLRConfig) -> NoamLR:
        return NoamLR(self._optimizer, config.num_warmup_steps)


@final
class _PolynomialDecayLRFactory:
    def __init__(self, optimizer: Optimizer, regime_section: RegimeSection) -> None:
        self._optimizer = optimizer
        self._regime_section = regime_section

    def create(self, config: PolynomialDecayLRConfig) -> PolynomialDecayLR:
        num_steps = self._regime_section.num_steps
        if num_steps is None:
            raise ValidationError(
                f"`regime.num_steps` must be specified when `lr_scheduler` is '{POLYNOMIAL_DECAY_LR}'."
            )

        if config.num_warmup_steps >= num_steps:
            raise ValidationError(
                f"`num_warmup_steps` must be less than `regime.warmup_steps` ({num_steps}), but is {config.num_warmup_steps} instead.", field="lr_scheduler.config"  # fmt: skip
            )

        num_param_groups = len(self._optimizer.param_groups)

        start_lrs = config.start_lr

        if isinstance(start_lrs, float):
            start_lrs = [start_lrs] * num_param_groups
        else:
            maybe_raise_param_group_length_error(
                "start_lr", start_lrs, num_param_groups
            )

        final_lrs = config.final_lr

        if isinstance(final_lrs, float):
            final_lrs = [final_lrs] * num_param_groups
        else:
            maybe_raise_param_group_length_error(
                "final_lr", final_lrs, num_param_groups
            )

        return PolynomialDecayLR(
            self._optimizer,
            num_steps,
            config.num_warmup_steps,
            power=config.power,
            start_lr=start_lrs,
            final_lr=final_lrs,
        )


@final
class _TriStageLRFactory:
    def __init__(self, optimizer: Optimizer, regime_section: RegimeSection) -> None:
        self._optimizer = optimizer
        self._regime_section = regime_section

    def create(self, config: TriStageLRConfig) -> TriStageLR:
        num_steps = self._regime_section.num_steps
        if num_steps is None:
            raise ValidationError(
                f"`regime.num_steps` must be specified when `lr_scheduler` is '{TRI_STAGE_LR}'."
            )

        num_param_groups = len(self._optimizer.param_groups)

        start_lr_scales = config.start_lr_scale

        if isinstance(start_lr_scales, float):
            start_lr_scales = [start_lr_scales] * num_param_groups
        else:
            maybe_raise_param_group_length_error(
                "start_lr_scale", start_lr_scales, num_param_groups
            )

        final_lr_scales = config.final_lr_scale

        if isinstance(final_lr_scales, float):
            final_lr_scales = [final_lr_scales] * num_param_groups
        else:
            maybe_raise_param_group_length_error(
                "final_lr_scale", final_lr_scales, num_param_groups
            )

        return TriStageLR(
            self._optimizer,
            num_steps,
            config.stage_ratio,
            start_lr_scale=start_lr_scales,
            final_lr_scale=final_lr_scales,
        )
