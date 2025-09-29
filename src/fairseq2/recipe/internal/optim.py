# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch.optim import Adafactor, AdamW, Optimizer

from fairseq2.gang import Gangs
from fairseq2.recipe.component import ComponentManager, ComponentNotKnownError
from fairseq2.recipe.config import AdafactorConfig, AdamWConfig, OptimizerSection
from fairseq2.recipe.error import OptimizerNotKnownError
from fairseq2.recipe.model import RecipeModel
from fairseq2.recipe.optim import prepare_parameter_groups


@final
class _RecipeOptimizerFactory:
    def __init__(
        self,
        section: OptimizerSection,
        component_manager: ComponentManager,
        gangs: Gangs,
    ) -> None:
        self._section = section
        self._component_manager = component_manager
        self._gangs = gangs

    def create(self) -> Optimizer:
        section = self._section

        try:
            return self._component_manager.create_component(
                Optimizer, section.name, section.config
            )
        except ComponentNotKnownError:
            raise OptimizerNotKnownError(section.name) from None


@final
class _AdamWFactory:
    def __init__(self, model: RecipeModel) -> None:
        self._model = model

    def create(self, config: AdamWConfig) -> Optimizer:
        parameters = prepare_parameter_groups(self._model, config.groups)

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


@final
class _AdafactorFactory:
    def __init__(self, model: RecipeModel) -> None:
        self._model = model

    def create(self, config: AdafactorConfig) -> Optimizer:
        parameters = prepare_parameter_groups(self._model, config.groups)

        return Adafactor(
            parameters,
            lr=config.lr,
            beta2_decay=config.beta2_decay,
            eps=config.eps,
            d=config.d,
            weight_decay=config.weight_decay,
            foreach=config.foreach,
            maximize=config.maximize,
        )
