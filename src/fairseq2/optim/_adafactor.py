# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional, Tuple, final

from torch.optim import Adafactor, Optimizer
from typing_extensions import override

from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

# isort: split

from fairseq2.optim._handler import OptimizerHandler
from fairseq2.optim._optimizer import ParameterCollection

ADAFACTOR_OPTIMIZER: Final = "adafactor"


@dataclass(kw_only=True)
class AdafactorConfig:
    """
    Adafactor is an optimizer that saves memory compared to AdamW
    by using low-rank representation of gradient running averages.
    It is recommended to use higher learning rate with it.
    """

    lr: float = 1e-2
    """The learning rate."""

    beta2_decay: float = -0.8
    """the decay rate of beta2.
    beta2 standardly refers to the coefficient used for computing the running average of the gradient squared."""

    eps: Tuple[Optional[float], float] = (None, 0.001)
    """epsilon1 is the term added to the denominator of the update calculation to improve numerical stability.
    epsilon2 is the term used to avoid having too small a weight update when applying parameter scaling.."""

    d: float = 1.0
    """the clipping threshold, used to avoid larger-than-desired updates."""

    weight_decay: float = 0.0
    """The weight decay coefficient."""

    foreach: Optional[bool] = None
    """whether foreach implementation of optimizer is used."""

    maximize: bool = False
    """If ``True``, maximizes the parameters instead of minimizing."""


@final
class AdafactorHandler(OptimizerHandler):
    @override
    def create(self, params: ParameterCollection, config: object) -> Optimizer:
        config = structure(config, AdafactorConfig)

        validate(config)

        return Adafactor(
            params,
            lr=config.lr,
            beta2_decay=config.beta2_decay,
            eps=config.eps,
            d=config.d,
            weight_decay=config.weight_decay,
            foreach=config.foreach,
            maximize=config.maximize,
        )

    @property
    @override
    def name(self) -> str:
        return ADAFACTOR_OPTIMIZER

    @property
    @override
    def config_kls(self) -> type[object]:
        return AdafactorConfig
