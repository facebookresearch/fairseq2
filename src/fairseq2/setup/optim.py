# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWHandler, OptimizerHandler
from fairseq2.optim.lr_scheduler import (
    COSINE_ANNEALING_LR,
    MYLE_LR,
    NOAM_LR,
    POLYNOMIAL_DECAY_LR,
    TRI_STAGE_LR,
    CosineAnnealingLRHandler,
    LRSchedulerHandler,
    MyleLRHandler,
    NoamLRHandler,
    PolynomialDecayLRHandler,
    TriStageLRHandler,
)


def _register_optimizers(context: RuntimeContext) -> None:
    registry = context.get_registry(OptimizerHandler)

    registry.register(ADAMW_OPTIMIZER, AdamWHandler())


def _register_lr_schedulers(context: RuntimeContext) -> None:
    registry = context.get_registry(LRSchedulerHandler)

    registry.register(COSINE_ANNEALING_LR, CosineAnnealingLRHandler())
    registry.register(MYLE_LR, MyleLRHandler())
    registry.register(NOAM_LR, NoamLRHandler())
    registry.register(POLYNOMIAL_DECAY_LR, PolynomialDecayLRHandler())
    registry.register(TRI_STAGE_LR, TriStageLRHandler())
