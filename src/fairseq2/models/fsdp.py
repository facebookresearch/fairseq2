# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module

from fairseq2.nn.data_parallel import FsdpGranularity, FsdpWrapper
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder


def apply_default_fsdp(
    model: Module, granularity: FsdpGranularity, wrapper: FsdpWrapper
) -> Module:
    if granularity == "model":
        return wrapper(model, reshard_after_forward=False)

    children = list(model.named_children())

    for name, child in children:
        if isinstance(child, (TransformerEncoder, TransformerDecoder)):
            if granularity == "stack":
                model.register_module(name, wrapper(child))
            else:
                layers = list(child.layers.named_children())

                for idx, (layer_name, layer) in enumerate(layers):
                    # We don't need to reshard the last layer since we will
                    # immediately gather it for the backward pass.
                    if idx < len(layers) - 1:
                        reshard_after_forward = None
                    else:
                        reshard_after_forward = False

                    child.layers.register_module(
                        layer_name, wrapper(layer, reshard_after_forward)
                    )

    return wrapper(model, reshard_after_forward=False)
