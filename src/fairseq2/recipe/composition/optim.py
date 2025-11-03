# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.optim import Optimizer

from fairseq2.recipe.component import register_component
from fairseq2.recipe.config import (
    ADAFACTOR_OPTIMIZER,
    ADAMW_OPTIMIZER,
    AdafactorConfig,
    AdamWConfig,
)
from fairseq2.recipe.internal.optim import (
    _AdafactorFactory,
    _AdamWFactory,
    _OptimizerFactory,
)
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def _register_optim(container: DependencyContainer) -> None:
    # Optimizer
    def get_optimizer(resolver: DependencyResolver) -> Optimizer:
        optimizer_factory = resolver.resolve(_OptimizerFactory)

        return optimizer_factory.create()

    container.register(Optimizer, get_optimizer, singleton=True)

    container.register_type(_OptimizerFactory)

    # AdamW
    def create_adamw(resolver: DependencyResolver, config: AdamWConfig) -> Optimizer:
        optimizer_factory = resolver.resolve(_AdamWFactory)

        return optimizer_factory.create(config)

    register_component(
        container,
        Optimizer,
        ADAMW_OPTIMIZER,
        config_kls=AdamWConfig,
        factory=create_adamw,
    )

    container.register_type(_AdamWFactory)

    # Adafactor
    def create_adafactor(
        resolver: DependencyResolver, config: AdafactorConfig
    ) -> Optimizer:
        optimizer_factory = resolver.resolve(_AdafactorFactory)

        return optimizer_factory.create(config)

    register_component(
        container,
        Optimizer,
        ADAFACTOR_OPTIMIZER,
        config_kls=AdafactorConfig,
        factory=create_adafactor,
    )

    container.register_type(_AdafactorFactory)
