# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import TypeVar

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
)
from torch.nn import Module

from fairseq2.nn import LayerStack
from fairseq2.utils.version import torch_greater_or_equal

ModelT_contra = TypeVar("ModelT_contra", bound=Module, contravariant=True)


def apply_default_activation_checkpointing(
    model: ModelT_contra, *, every_nth_layer: int = 1
) -> None:
    applied = _do_apply_ac(model, every_nth_layer)

    if not applied:
        raise ValueError("`model` must contain at least one layer stack.")


def _do_apply_ac(module: Module, every_nth_layer: int) -> bool:
    applied = False

    children = list(module.named_children())

    for child_name, child in children:
        if isinstance(child, LayerStack):
            _do_apply_layerwise_ac(child, every_nth_layer)

            applied = True
        else:
            if _do_apply_ac(child, every_nth_layer):
                applied = True

    return applied


def _do_apply_layerwise_ac(stack: LayerStack, every_nth_layer: int) -> None:
    if not torch_greater_or_equal(2, 6):
        warnings.filterwarnings(
            action="ignore", message=r".*`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated.*"  # fmt: skip
        )

    layers = list(stack.layers.named_children())

    for layer_idx, (layer_name, layer) in enumerate(layers):
        if layer_idx % every_nth_layer == 0:
            wrapper = CheckpointWrapper(
                layer,
                CheckpointImpl.NO_REENTRANT,
                preserve_rng_state=True,
            )

            stack.layers.register_module(layer_name, wrapper)
