# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.optim import Optimizer

from fairseq2.dependency import DependencyResolver
from fairseq2.error import SetupError
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
from fairseq2.recipe.component import resolve_component
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
from fairseq2.recipe.error import UnspecifiedNumberOfStepsError
from fairseq2.utils.structured import StructureError


def create_lr_scheduler(resolver: DependencyResolver) -> LRScheduler:
    section = get_config_section(resolver, "lr_scheduler", LRSchedulerSection)

    if section.name is None:
        optimizer = resolver.resolve(Optimizer)

        return NoopLR(optimizer)

    try:
        return resolve_component(resolver, LRScheduler, section.name, section.config)
    except StructureError as ex:
        raise StructureError(
            f"The '{section.name}' learning rate scheduler configuration cannot be parsed. See the nested exception for details."
        ) from ex


def create_cosine_annealing_lr(
    resolver: DependencyResolver, config: CosineAnnealingLRConfig
) -> LRScheduler:
    regime_section = get_config_section(resolver, "regime", RegimeSection)

    optimizer = resolver.resolve(Optimizer)

    if config.cycle_len is None:
        num_steps = regime_section.num_steps
        if num_steps is None:
            raise UnspecifiedNumberOfStepsError(COSINE_ANNEALING_LR)

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
        raise SetupError(
            "The optimizer does not have a parameter group with an assigned learning rate."
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


def create_myle_lr(resolver: DependencyResolver, config: MyleLRConfig) -> LRScheduler:
    optimizer = resolver.resolve(Optimizer)

    return MyleLR(optimizer, config.num_warmup_steps, start_lr=config.start_lr)


def create_noam_lr(resolver: DependencyResolver, config: NoamLRConfig) -> LRScheduler:
    optimizer = resolver.resolve(Optimizer)

    return NoamLR(optimizer, config.num_warmup_steps)


def create_polynomial_decay_lr(
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


def create_tri_stage_lr(
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
