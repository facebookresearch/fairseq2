# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.optim import Optimizer

from fairseq2.context import get_runtime_context
from fairseq2.optim.lr_scheduler.handler import (
    LRSchedulerHandler,
    LRSchedulerNotFoundError,
)
from fairseq2.optim.lr_scheduler.lr_scheduler import LRScheduler
from fairseq2.utils.config import process_config
from fairseq2.utils.structured import structure


def create_lr_scheduler(
    name: str,
    optimizer: Optimizer,
    config: object = None,
    *,
    max_num_steps: int | None = None,
) -> LRScheduler:
    context = get_runtime_context()

    registry = context.get_registry(LRSchedulerHandler)

    try:
        handler = registry.get(name)
    except LookupError:
        raise LRSchedulerNotFoundError(name) from None

    if config is None:
        try:
            config = handler.config_kls()
        except TypeError:
            raise ValueError(
                f"`config` must be specified for the '{name}' learning rate scheduler."
            ) from None
    else:
        config = structure(config, handler.config_kls)

    process_config(config)

    return handler.create(optimizer, config, max_num_steps)
