# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
)
from torch.nn import ModuleList


def apply_layerwise_ac(stack: ModuleList, every_nth_layer: int) -> None:
    layers = list(stack)

    for idx, layer in enumerate(layers):
        if idx % every_nth_layer == 0:
            stack[idx] = CheckpointWrapper(
                layer, CheckpointImpl.NO_REENTRANT, preserve_rng_state=True
            )
