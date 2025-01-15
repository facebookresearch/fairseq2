# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module
from torch.optim import Optimizer

from fairseq2.context import get_runtime_context
from fairseq2.optim._handler import OptimizerHandler, OptimizerNotFoundError
from fairseq2.optim._optimizer import ParameterCollection
from fairseq2.utils.config import process_config
from fairseq2.utils.structured import structure


def create_optimizer(
    name: str, params: ParameterCollection | Module, config: object = None
) -> Optimizer:
    context = get_runtime_context()

    registry = context.get_registry(OptimizerHandler)

    try:
        handler = registry.get(name)
    except LookupError:
        raise OptimizerNotFoundError(name) from None

    if config is None:
        try:
            config = handler.config_kls()
        except TypeError:
            raise ValueError(
                f"`config` must be specified for the '{name}' optimizer."
            ) from None
    else:
        config = structure(config, handler.config_kls)

    process_config(config)

    if isinstance(params, Module):
        params = params.parameters()

    return handler.create(params, config)
