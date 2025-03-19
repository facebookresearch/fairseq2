# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch.optim import Optimizer

from fairseq2.context import RuntimeContext
from fairseq2.optim import OptimizerHandler, UnknownOptimizerError
from fairseq2.optim.lr_scheduler import (
    LRScheduler,
    LRSchedulerHandler,
    NoopLR,
    UnknownLRSchedulerError,
)
from fairseq2.recipes import Model
from fairseq2.recipes.config import (
    LRSchedulerSection,
    OptimizerSection,
    RegimeSection,
    get_config_section,
)
from fairseq2.registry import Provider
from fairseq2.utils.structured import StructureError


def create_optimizer(
    context: RuntimeContext, recipe_config: object, model: Model
) -> Optimizer:
    optimizer_handlers = context.get_registry(OptimizerHandler)

    creator = OptimizerCreator(optimizer_handlers)

    return creator.create(recipe_config, model)


@final
class OptimizerCreator:
    _optimizer_handlers: Provider[OptimizerHandler]

    def __init__(self, optimizer_handlers: Provider[OptimizerHandler]) -> None:
        self._optimizer_handlers = optimizer_handlers

    def create(self, recipe_config: object, model: Model) -> Optimizer:
        optimizer_section = get_config_section(
            recipe_config, "optimizer", OptimizerSection
        )

        try:
            handler = self._optimizer_handlers.get(optimizer_section.name)
        except LookupError:
            raise UnknownOptimizerError(optimizer_section.name) from None

        params = model.module.parameters()

        try:
            return handler.create(params, optimizer_section.config)
        except StructureError as ex:
            raise StructureError(
                "`optimizer.config` cannot be structured. See the nested exception for details."
            ) from ex


def create_lr_scheduler(
    context: RuntimeContext, recipe_config: object, optimizer: Optimizer
) -> LRScheduler:
    lr_scheduler_handlers = context.get_registry(LRSchedulerHandler)

    creator = LRSchedulerCreator(lr_scheduler_handlers)

    return creator.create(recipe_config, optimizer)


@final
class LRSchedulerCreator:
    _lr_scheduler_handlers: Provider[LRSchedulerHandler]

    def __init__(self, lr_scheduler_handlers: Provider[LRSchedulerHandler]) -> None:
        self._lr_scheduler_handlers = lr_scheduler_handlers

    def create(self, recipe_config: object, optimizer: Optimizer) -> LRScheduler:
        lr_scheduler_section = get_config_section(
            recipe_config, "lr_scheduler", LRSchedulerSection
        )

        if lr_scheduler_section.name is None:
            return NoopLR(optimizer)

        try:
            handler = self._lr_scheduler_handlers.get(lr_scheduler_section.name)
        except LookupError:
            raise UnknownLRSchedulerError(lr_scheduler_section.name) from None

        regime_section = get_config_section(recipe_config, "regime", RegimeSection)

        try:
            return handler.create(
                optimizer, lr_scheduler_section.config, regime_section.num_steps
            )
        except StructureError as ex:
            raise StructureError(
                "`lr_scheduler.config` cannot be structured. See the nested exception for details."
            ) from ex
