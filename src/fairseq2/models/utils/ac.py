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
from torch.nn import Module

from fairseq2.nn import LayerStack


def apply_layerwise_activation_checkpointing(
    model: Module, *, every_nth_layer: int = 1
) -> None:
    _do_apply_layerwise_ac(model, every_nth_layer)


def _do_apply_layerwise_ac(module: Module, every_nth_layer: int) -> None:
    children = list(module.named_children())

    for child_name, child in children:
        if isinstance(child, LayerStack):
            _apply_ac_to_stack(child, every_nth_layer)
        else:
            _do_apply_layerwise_ac(child, every_nth_layer)


def _apply_ac_to_stack(stack: LayerStack, every_nth_layer: int) -> None:
    layers = list(stack.layers.named_children())

    for idx, (layer_name, layer) in enumerate(layers):
        if idx % every_nth_layer == 0:
            wrapper = CheckpointWrapper(
                layer, CheckpointImpl.NO_REENTRANT, preserve_rng_state=True
            )

            stack.layers.register_module(layer_name, wrapper)
