# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from torch.nn import Module
from torch.optim import Optimizer

from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.optim.adamw import AdamW
from fairseq2.optim.optimizer import ParameterCollection

optimizer_factories = ConfigBoundFactoryRegistry[[ParameterCollection], Optimizer]()

optimizer_factory = optimizer_factories.decorator


def create_optimizer(
    name: str,
    params: Union[Module, ParameterCollection],
    unstructured_config: object = None,
) -> Optimizer:
    """Create an optimizer of type registered with ``name``.

    :param name:
        The name of the optimizer.
    :param params:
        The parameters or :class:`Module` to optimize.
    :param unstructured_config:
        The unstructured configuration of the optimizer.
    """
    factory = optimizer_factories.get(name, unstructured_config)

    if isinstance(params, Module):
        params = params.parameters()

    return factory(params)


@dataclass(kw_only=True)
class AdamWConfig:
    """Holds the configuration of a :class:`AdamW`."""

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

    use_fp32: bool = False
    """If ``True``, stores the optimizer state in single precision and converts
    gradients on-the-fly to single precision for numerical stability."""


@optimizer_factory("adamw")
def create_adamw_optimizer(config: AdamWConfig, params: ParameterCollection) -> AdamW:
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
        impl=config.impl,
        use_fp32=config.use_fp32,
    )
