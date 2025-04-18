# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
)
from torch.nn import Module

from fairseq2.nn import LayerStack
from fairseq2.utils.version import torch_greater_or_equal


def use_layerwise_activation_checkpointing(
    model: Module, preserve_rng_state: bool = True
) -> None:
    """
    Uses layer-wise activation checkpointing in ``model``.

    :param module: The module to apply checkpointing.
    :param preserve_rng_state: If ``True``, stashes the states of the default
        random number generators for the CPU and the device of ``module`` during
        the original forward pass and restores them during the recomputation.
    """
    if not torch_greater_or_equal(2, 6):
        warnings.filterwarnings(
            action="ignore", message=r".*`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated.*"  # fmt: skip
        )

    children = list(model.named_children())

    for name, child in children:
        if isinstance(child, LayerStack):
            layers = list(child.layers.named_children())

            for layer_name, layer in layers:
                wrapper = CheckpointWrapper(
                    layer,
                    CheckpointImpl.NO_REENTRANT,
                    preserve_rng_state=preserve_rng_state,
                )

                child.layers.register_module(layer_name, wrapper)
