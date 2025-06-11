# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TypeVar

from torch.nn import Module

from fairseq2.nn import LayerStack
from fairseq2.nn.data_parallel import FSDPGranularity, FSDPWrapper

ModelT_contra = TypeVar("ModelT_contra", bound=Module, contravariant=True)


def apply_default_fsdp(
    model: ModelT_contra, granularity: FSDPGranularity, wrapper: FSDPWrapper
) -> None:
    applied = _do_apply_default_fsdp(model, granularity, wrapper)

    if not applied:
        raise ValueError("`model` must contain at least one layer stack.")


def _do_apply_default_fsdp(
    module: Module, granularity: FSDPGranularity, wrapper: FSDPWrapper
) -> bool:
    applied = False

    children = list(module.named_children())

    for child_name, child in children:
        if isinstance(child, LayerStack):
            if granularity == "stack":
                child = wrapper(child)

                module.register_module(child_name, child)

                applied = True

                continue

            if granularity == "layer":
                _do_apply_layerwise_fsdp(child, granularity, wrapper)

                applied = True

                continue

            raise ValueError(
                f"`granularity` must be 'layer' or 'stack', but is '{granularity}' instead."
            )
        else:
            if _do_apply_default_fsdp(child, granularity, wrapper):
                applied = True

    return applied


def _do_apply_layerwise_fsdp(
    stack: LayerStack, granularity: FSDPGranularity, wrapper: FSDPWrapper
) -> None:
    layers = list(stack.layers.named_children())

    for layer_idx, (layer_name, layer) in enumerate(layers):
        # We don't need to reshard the last layer since we will immediately
        # gather it for the backward pass.
        wrapped = wrapper(
            layer, reshard_after_forward=None if layer_idx < len(layers) - 1 else False
        )

        stack.layers.register_module(layer_name, wrapped)
