# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.optim import AdamW, Optimizer

from fairseq2.dependency import DependencyResolver
from fairseq2.recipe.component import resolve_component
from fairseq2.recipe.config import (
    AdamWConfig,
    ComponentSection,
    get_recipe_config_section,
)
from fairseq2.recipe.model import Model


def create_optimizer(resolver: DependencyResolver) -> Optimizer:
    section = get_recipe_config_section(resolver, "optimizer", ComponentSection)

    return resolve_component(resolver, Optimizer, section.name, section.config)


def create_adamw(resolver: DependencyResolver, config: AdamWConfig) -> Optimizer:
    model = resolver.resolve(Model)

    parameters = model.module.parameters()

    kwargs = {}

    impl = config.impl
    if impl != "auto":
        if impl == "naive":
            # Disables both 'foreach' and 'fused'.
            kwargs["foreach"] = False
        else:
            kwargs[impl] = True

    return AdamW(
        parameters,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
        maximize=config.maximize,
        capturable=config.capturable,
        differentiable=config.differentiable,
        **kwargs,
    )
