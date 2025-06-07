# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module

from fairseq2.nn import LayerStack
from fairseq2.nn.data_parallel import FSDPGranularity, FSDPWrapper


def apply_layerwise_fsdp(
    model: Module, granularity: FSDPGranularity, wrapper: FSDPWrapper
) -> None:
    _do_apply_layerwise_fsdp(model, granularity, wrapper)


def _do_apply_layerwise_fsdp(
    module: Module, granularity: FSDPGranularity, wrapper: FSDPWrapper
) -> None:
    children = list(module.named_children())

    for child_name, child in children:
        if isinstance(child, LayerStack):
            if granularity == "stack":
                child = wrapper(child)

                module.register_module(child_name, child)

                continue

            if granularity == "layer":
                _apply_fsdp_to_stack(child, granularity, wrapper)

                continue

            raise ValueError(
                f"`granularity` must be 'layer' or 'stack', but is '{granularity}' instead."
            )
        else:
            _do_apply_layerwise_fsdp(child, granularity, wrapper)


def _apply_fsdp_to_stack(
    stack: LayerStack, granularity: FSDPGranularity, wrapper: FSDPWrapper
) -> None:
    layers = list(stack.layers.named_children())

    for idx, (layer_name, layer) in enumerate(layers):
        # We don't need to reshard the last layer since we will immediately
        # gather it for the backward pass.
        wrapped = wrapper(
            layer, reshard_after_forward=None if idx < len(layers) - 1 else False
        )

        stack.layers.register_module(layer_name, wrapped)
