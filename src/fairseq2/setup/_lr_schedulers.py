# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.optim.lr_scheduler import (
    CosineAnnealingLRHandler,
    LRSchedulerHandler,
    MyleLRHandler,
    NoamLRHandler,
    PolynomialDecayLRHandler,
    TriStageLRHandler,
)


def _register_lr_schedulers(context: RuntimeContext) -> None:
    registry = context.get_registry(LRSchedulerHandler)

    handler: LRSchedulerHandler

    # Cosine Annealing
    handler = CosineAnnealingLRHandler()

    registry.register(handler.name, handler)

    # Myle
    handler = MyleLRHandler()

    registry.register(handler.name, handler)

    # Noam
    handler = NoamLRHandler()

    registry.register(handler.name, handler)

    # Polynomial Decay
    handler = PolynomialDecayLRHandler()

    registry.register(handler.name, handler)

    # Tri-Stage
    handler = TriStageLRHandler()

    registry.register(handler.name, handler)
