# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWHandler, OptimizerHandler


def _register_optimizers(context: RuntimeContext) -> None:
    registry = context.get_registry(OptimizerHandler)

    registry.register(ADAMW_OPTIMIZER, AdamWHandler())
