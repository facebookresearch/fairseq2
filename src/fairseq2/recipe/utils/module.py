# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from torch import Tensor
from torch.nn import Module

from fairseq2.utils.version import torch_greater_or_equal


@dataclass(kw_only=True)
class ModuleSizeInfo:
    """Holds the size information of a module."""

    param_size: int = 0
    """The total size of all parameters."""

    param_size_bytes: int = 0
    """The total size of all parameters, in bytes."""

    trainable_param_size: int = 0
    """The total size of all trainable parameters."""

    trainable_param_size_bytes: int = 0
    """The total size of all trainable parameters, in bytes."""

    buffer_size: int = 0
    """The total size of all buffers."""

    buffer_size_bytes: int = 0
    """The total size of all buffers, in bytes."""

    total_size: int = 0
    """The total size of the module."""

    total_size_bytes: int = 0
    """The total size of the module, in bytes."""


def get_module_size_info(module: Module) -> ModuleSizeInfo:
    """Return the size information of ``module`` and its descendant modules."""

    def get_numel(tensor: Tensor) -> int:
        if torch_greater_or_equal(2, 6):
            from torch.distributed.tensor import DTensor

            if isinstance(tensor, DTensor):
                return cast(DTensor, tensor.detach()).to_local().numel()  # type: ignore[no-any-return]

        return tensor.numel()

    info = ModuleSizeInfo()

    param: Tensor | None

    for param in module.parameters():
        if param is None:
            continue

        numel = get_numel(param)

        size_bytes = numel * param.element_size()

        info.param_size += numel
        info.param_size_bytes += size_bytes

        if param.requires_grad:
            info.trainable_param_size += numel
            info.trainable_param_size_bytes += size_bytes

        info.total_size += numel
        info.total_size_bytes += size_bytes

    for buffer in module.buffers():
        if buffer is None:
            continue

        numel = buffer.numel()

        size_bytes = numel * buffer.element_size()

        info.buffer_size += numel
        info.buffer_size_bytes += size_bytes

        info.total_size += numel
        info.total_size_bytes += size_bytes

    return info
