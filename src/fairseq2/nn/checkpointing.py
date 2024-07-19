# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.nn import Module

from fairseq2.nn.transformer import TransformerDecoderLayer, TransformerEncoderLayer


def use_layerwise_activation_checkpointing(
    module: Module, preserve_rng_state: bool = True
) -> None:
    """Use layer-wise activation checkpointing in ``module``.

    :param module:
        The module to apply checkpointing.
    :param preserve_rng_state:
        If ``True``, stashes the states of the default random number generators
        for the CPU and the device of ``module`` during the original forward
        pass and restores them during the recomputation.
    """
    wrap = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=preserve_rng_state,
    )

    def check_module_type(m: Module) -> bool:
        return isinstance(m, (TransformerEncoderLayer, TransformerDecoderLayer))

    apply_activation_checkpointing(
        module, checkpoint_wrapper_fn=wrap, check_fn=check_module_type
    )
