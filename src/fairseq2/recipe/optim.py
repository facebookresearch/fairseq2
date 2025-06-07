# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.optim import AdamW, Optimizer

from fairseq2.model.context import ModelContext
from fairseq2.recipe.component import ComponentManager, UnknownComponentError
from fairseq2.recipe.config import AdamWConfig, OptimizerSection, get_config_section
from fairseq2.recipe.error import UnknownOptimizerError
from fairseq2.runtime.dependency import DependencyResolver


def _create_optimizer(resolver: DependencyResolver) -> Optimizer:
    section = get_config_section(resolver, "optimizer", OptimizerSection)

    component_manager = resolver.resolve(ComponentManager)

    try:
        return component_manager.create_component(
            Optimizer, section.name, section.config
        )
    except UnknownComponentError:
        raise UnknownOptimizerError(section.name) from None


def _create_adamw(resolver: DependencyResolver, config: AdamWConfig) -> Optimizer:
    model_context = resolver.resolve(ModelContext)

    parameters = model_context.model.parameters()

    kwargs = {}

    impl = config.impl
    if impl != "auto":
        if impl == "naive":
            kwargs["foreach"] = False  # disables both 'foreach' and 'fused'.
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
