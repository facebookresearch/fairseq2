# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import NoneType

from torch.optim import Optimizer

from fairseq2.optim.lr_schedulers import LRScheduler, PassthroughLR
from fairseq2.recipe.component import register_component
from fairseq2.recipe.config import (
    COSINE_ANNEALING_LR,
    MYLE_LR,
    NOAM_LR,
    PASSTHROUGH_LR,
    POLYNOMIAL_DECAY_LR,
    TRI_STAGE_LR,
    CosineAnnealingLRConfig,
    MyleLRConfig,
    NoamLRConfig,
    PolynomialDecayLRConfig,
    TriStageLRConfig,
)
from fairseq2.recipe.internal.lr_schedulers import (
    _CosineAnnealingLRFactory,
    _MyleLRFactory,
    _NoamLRFactory,
    _PolynomialDecayLRFactory,
    _RecipeLRSchedulerFactory,
    _TriStageLRFactory,
)
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def _register_lr_schedulers(container: DependencyContainer) -> None:
    # LRScheduler
    def create_lr_scheduler(resolver: DependencyResolver) -> LRScheduler:
        lr_factory = resolver.resolve(_RecipeLRSchedulerFactory)

        return lr_factory.create()

    container.register(LRScheduler, create_lr_scheduler, singleton=True)

    container.register_type(_RecipeLRSchedulerFactory)

    # Passthrough
    def create_passthrough_lr(
        resolver: DependencyResolver, config: None
    ) -> LRScheduler:
        optimizer = resolver.resolve(Optimizer)

        return PassthroughLR(optimizer)

    register_component(
        container,
        LRScheduler,
        PASSTHROUGH_LR,
        config_kls=NoneType,
        factory=create_passthrough_lr,
    )

    # Cosine Annealing
    def create_cosine_annealing_lr(
        resolver: DependencyResolver, config: CosineAnnealingLRConfig
    ) -> LRScheduler:
        lr_factory = resolver.resolve(_CosineAnnealingLRFactory)

        return lr_factory.create(config)

    register_component(
        container,
        LRScheduler,
        COSINE_ANNEALING_LR,
        config_kls=CosineAnnealingLRConfig,
        factory=create_cosine_annealing_lr,
    )

    container.register_type(_CosineAnnealingLRFactory)

    # Myle
    def create_myle_lr(
        resolver: DependencyResolver, config: MyleLRConfig
    ) -> LRScheduler:
        lr_factory = resolver.resolve(_MyleLRFactory)

        return lr_factory.create(config)

    register_component(
        container,
        LRScheduler,
        MYLE_LR,
        config_kls=MyleLRConfig,
        factory=create_myle_lr,
    )

    container.register_type(_MyleLRFactory)

    # Noam
    def create_noam_lr(
        resolver: DependencyResolver, config: NoamLRConfig
    ) -> LRScheduler:
        lr_factory = resolver.resolve(_NoamLRFactory)

        return lr_factory.create(config)

    register_component(
        container,
        LRScheduler,
        NOAM_LR,
        config_kls=NoamLRConfig,
        factory=create_noam_lr,
    )

    container.register_type(_NoamLRFactory)

    # Polynomial Decay
    def create_polynomial_decay_lr(
        resolver: DependencyResolver, config: PolynomialDecayLRConfig
    ) -> LRScheduler:
        lr_factory = resolver.resolve(_PolynomialDecayLRFactory)

        return lr_factory.create(config)

    register_component(
        container,
        LRScheduler,
        POLYNOMIAL_DECAY_LR,
        config_kls=PolynomialDecayLRConfig,
        factory=create_polynomial_decay_lr,
    )

    container.register_type(_PolynomialDecayLRFactory)

    # Tri-Stage
    def create_tri_stage_lr(
        resolver: DependencyResolver, config: TriStageLRConfig
    ) -> LRScheduler:
        lr_factory = resolver.resolve(_TriStageLRFactory)

        return lr_factory.create(config)

    register_component(
        container,
        LRScheduler,
        TRI_STAGE_LR,
        config_kls=TriStageLRConfig,
        factory=create_tri_stage_lr,
    )

    container.register_type(_TriStageLRFactory)
