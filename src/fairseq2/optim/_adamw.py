# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, final

from torch.optim import AdamW, Optimizer
from typing_extensions import override

from fairseq2.optim._handler import OptimizerHandler
from fairseq2.optim._optimizer import ParameterCollection
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

ADAMW_OPTIMIZER: Final = "adamw"


@dataclass(kw_only=True)
class AdamWConfig:
    lr: float = 1e-3
    """The learning rate."""

    betas: tuple[float, float] = (0.9, 0.999)
    """The coefficients used for computing running averages of gradient and its
    square."""

    eps: float = 1e-8
    """The term added to the denominator to improve numerical stability."""

    weight_decay: float = 0.0
    """The weight decay coefficient."""

    amsgrad: bool = False
    """If ``True``, uses the AMSGrad variant."""

    maximize: bool = False
    """If ``True``, maximizes the parameters instead of minimizing."""

    capturable: bool = False
    """If ``True``, it is safe to capture this instance in a CUDA graph."""

    differentiable: bool = False
    """If ``True``, runs the optimizer step under autograd."""

    impl: Literal["auto", "foreach", "fused", "naive"] = "auto"
    """The implementation variant. See :class:`torch.optim.AdamW` for details."""


@final
class AdamWHandler(OptimizerHandler):
    @override
    def create(self, params: ParameterCollection, config: object) -> Optimizer:
        config = structure(config, AdamWConfig)

        validate(config)

        kwargs = {}

        impl = config.impl
        if impl != "auto":
            if impl == "naive":
                # Disables both 'foreach' and 'fused'.
                kwargs["foreach"] = False
            else:
                kwargs[impl] = True

        return AdamW(
            params,
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

    @property
    @override
    def name(self) -> str:
        return ADAMW_OPTIMIZER

    @property
    @override
    def config_kls(self) -> type[object]:
        return AdamWConfig
