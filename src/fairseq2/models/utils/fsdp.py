# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import ModuleList

from fairseq2.nn.fsdp import FSDPWrapper


def apply_layerwise_fsdp(stack: ModuleList, wrapper: FSDPWrapper) -> None:
    layers = list(stack)

    for idx, layer in enumerate(layers):
        # We don't need to reshard the last layer since we will immediately
        # gather it for the backward pass.
        reshard_after_forward = None if idx < len(layers) - 1 else False

        stack[idx] = wrapper(layer, reshard_after_forward=reshard_after_forward)
