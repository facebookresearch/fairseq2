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

    def create(self, config: CosineAnnealingLRConfig) -> LRScheduler:
        optimizer = self._optimizer

        # TODO: fix!
        if len(optimizer.param_groups) > 1:
            raise ValueError(
                "`optimizer` must not have more than one optimizer parameter group."
            )

        try:
            lr: float = optimizer.param_groups[0]["lr"]
        except KeyError:
            raise ValueError("`optimizer` must have a learning rate.") from None

        if config.cycle_len is None:
            num_steps = self._regime_section.num_steps
            if num_steps is None:
                raise ValidationError(
                    f"`regime.num_steps` must be specified when `lr_scheduler` is '{COSINE_ANNEALING_LR}' and `lr_scheduler.config.cycle_len` is not specified."
                )

            cycle_len = num_steps - config.num_warmup_steps
        else:
            cycle_len = config.cycle_len

        if config.final_lr is not None and config.final_lr_scale is not None:
            raise InternalError(
                "`config.final_lr` and `config.final_lr_scale` are both specified."
            )

        if config.final_lr is not None:
            final_lr = config.final_lr
        else:
            if config.final_lr_scale is None:
                raise InternalError(
                    "`config.final_lr` and `config.final_lr_scale` are both `None`."
                )

            final_lr = lr * config.final_lr_scale

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


@final
class _MyleLRFactory:
    def __init__(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    def create(self, config: MyleLRConfig) -> LRScheduler:
        return MyleLR(
            self._optimizer, config.num_warmup_steps, start_lr=config.start_lr
        )


@final
class _NoamLRFactory:
    def __init__(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    def create(self, config: NoamLRConfig) -> LRScheduler:
        return NoamLR(self._optimizer, config.num_warmup_steps)


@final
class _PolynomialDecayLRFactory:
    def __init__(self, optimizer: Optimizer, regime_section: RegimeSection) -> None:
        self._optimizer = optimizer
        self._regime_section = regime_section

    def create(self, config: PolynomialDecayLRConfig) -> LRScheduler:
        num_steps = self._regime_section.num_steps
        if num_steps is None:
            raise ValidationError(
                f"`regime.num_steps` must be specified when `lr_scheduler` is '{POLYNOMIAL_DECAY_LR}'."
            )

        return PolynomialDecayLR(
            self._optimizer,
            num_steps,
            config.num_warmup_steps,
            power=config.power,
            start_lr=config.start_lr,
            final_lr=config.final_lr,
        )


@final
class _TriStageLRFactory:
    def __init__(self, optimizer: Optimizer, regime_section: RegimeSection) -> None:
        self._optimizer = optimizer
        self._regime_section = regime_section

    def create(self, config: TriStageLRConfig) -> LRScheduler:
        num_steps = self._regime_section.num_steps
        if num_steps is None:
            raise ValidationError(
                f"`regime.num_steps` must be specified when `lr_scheduler` is '{TRI_STAGE_LR}'."
            )

        return TriStageLR(
            self._optimizer,
            num_steps,
            config.stage_ratio,
            start_lr_scale=config.start_lr_scale,
            final_lr_scale=config.final_lr_scale,
        )
