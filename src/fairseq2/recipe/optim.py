# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch
from torch.optim import Adafactor, AdamW, Optimizer

from fairseq2.gang import Gangs
from fairseq2.model import Model
from fairseq2.optim.dynamic_loss_scaler import supports_manual_grad_scaling
from fairseq2.recipe.component import ComponentManager, ComponentNotKnownError
from fairseq2.recipe.config import (
    AdafactorConfig,
    AdamWConfig,
    OptimizerNotKnownError,
    OptimizerSection,
    TrainerSection,
)


@final
class OptimizerFactory:
    def __init__(
        self,
        section: OptimizerSection,
        trainer_section: TrainerSection,
        component_manager: ComponentManager,
        gangs: Gangs,
    ) -> None:
        self._section = section
        self._trainer_section = trainer_section
        self._component_manager = component_manager
        self._gangs = gangs

    def create(self) -> Optimizer:
        section = self._section

        try:
            optimizer = self._component_manager.create_component(
                Optimizer, section.name, section.config
            )
        except ComponentNotKnownError:
            raise OptimizerNotKnownError(section.name) from None

        if self._trainer_section.dtype == torch.float16 and self._gangs.sdp.size > 1:
            if not supports_manual_grad_scaling(optimizer):
                raise ManualGradScalingNotSupportedError(section.name)

        return optimizer


class ManualGradScalingNotSupportedError(Exception):
    def __init__(self, optimizer_name: str) -> None:
        super().__init__(
            f"{optimizer_name} optimizer does not support manual fp16 gradient scaling which is required for FSDP."
        )

        self.optimizer_name = optimizer_name


@final
class AdamWFactory:
    def __init__(self, model: Model) -> None:
        self._model = model

    def create(self, config: AdamWConfig) -> Optimizer:
        parameters = self._model.module.parameters()

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
class AdafactorFactory:
    def __init__(self, model: Model) -> None:
        self._model = model

    def create(self, config: AdafactorConfig) -> Optimizer:
        parameters = self._model.module.parameters()

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
