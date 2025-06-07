# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.optim import Optimizer

from fairseq2.error import InternalError, NotSupportedError
from fairseq2.logging import log
from fairseq2.optim.lr_schedulers import (
    CosineAnnealingLR,
    LRScheduler,
    MyleLR,
    NoamLR,
    NoopLR,
    PolynomialDecayLR,
    TriStageLR,
)
from fairseq2.recipe.component import ComponentManager, UnknownComponentError
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
    get_config_section,
)
from fairseq2.recipe.error import UnknownLRSchedulerError, UnspecifiedNumberOfStepsError
from fairseq2.runtime.dependency import DependencyResolver


def _create_lr_scheduler(resolver: DependencyResolver) -> LRScheduler:
    section = get_config_section(
        resolver, "lr_scheduler", LRSchedulerSection, allow_none=True
    )

    if section is None:
        optimizer = resolver.resolve(Optimizer)

        return NoopLR(optimizer)

    component_manager = resolver.resolve(ComponentManager)

    try:
        return component_manager.create_component(
            LRScheduler, section.name, section.config
        )
    except UnknownComponentError:
        raise UnknownLRSchedulerError(section.name) from None


def _create_cosine_annealing_lr(
    resolver: DependencyResolver, config: CosineAnnealingLRConfig
) -> LRScheduler:
    regime_section = get_config_section(resolver, "regime", RegimeSection)

    optimizer = resolver.resolve(Optimizer)

    if len(optimizer.param_groups) > 1:
        raise NotSupportedError(
            f"'{COSINE_ANNEALING_LR}' does not support more than one optimizer parameter group."
        )

    try:
        lr: float = optimizer.param_groups[0]["lr"]
    except KeyError:
        raise NotSupportedError(
            "The optimizer does not have a learning rate."
        ) from None

    if config.cycle_len is None:
        num_steps = regime_section.num_steps
        if num_steps is None:
            raise UnspecifiedNumberOfStepsError(COSINE_ANNEALING_LR)

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


def _create_myle_lr(resolver: DependencyResolver, config: MyleLRConfig) -> LRScheduler:
    optimizer = resolver.resolve(Optimizer)

    return MyleLR(optimizer, config.num_warmup_steps, start_lr=config.start_lr)


def _create_noam_lr(resolver: DependencyResolver, config: NoamLRConfig) -> LRScheduler:
    optimizer = resolver.resolve(Optimizer)

    return NoamLR(optimizer, config.num_warmup_steps)


def _create_polynomial_decay_lr(
    resolver: DependencyResolver, config: PolynomialDecayLRConfig
) -> LRScheduler:
    regime_section = get_config_section(resolver, "regime", RegimeSection)

    optimizer = resolver.resolve(Optimizer)

    num_steps = regime_section.num_steps
    if num_steps is None:
        raise UnspecifiedNumberOfStepsError(POLYNOMIAL_DECAY_LR)

    return PolynomialDecayLR(
        optimizer,
        num_steps,
        config.num_warmup_steps,
        power=config.power,
        start_lr=config.start_lr,
        final_lr=config.final_lr,
    )


def _create_tri_stage_lr(
    resolver: DependencyResolver, config: TriStageLRConfig
) -> LRScheduler:
    regime_section = get_config_section(resolver, "regime", RegimeSection)

    optimizer = resolver.resolve(Optimizer)

    num_steps = regime_section.num_steps
    if num_steps is None:
        raise UnspecifiedNumberOfStepsError(TRI_STAGE_LR)

    return TriStageLR(
        optimizer,
        num_steps,
        config.stage_ratio,
        start_lr_scale=config.start_lr_scale,
        final_lr_scale=config.final_lr_scale,
    )
