# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal, Protocol, TypeAlias, final

from torch import Tensor
from torch.distributed._shard import load_with_process_group
from torch.nn import Module

from fairseq2.error import NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.nn.utils.module import (
    reset_non_persistent_buffers,
    reset_parameters,
    to_empty,
)
from fairseq2.typing import ContextManager, Device

FsdpGranularity: TypeAlias = Literal["layer", "stack", "model"]


class FsdpWrapper(Protocol):
    def __call__(
        self, module: Module, reshard_after_forward: bool | None = None
    ) -> Module: ...


class FsdpApplier(Protocol):
    def __call__(
        self, module: Module, granularity: FsdpGranularity, wrapper: FsdpWrapper
    ) -> Module: ...


@final
class FsdpParameterInitializer:
    """Initializes the parameters and buffers of an FSDP module.

    This is a convenience callable to pass to the ``param_init_fn`` parameter of
    :class:`FSDP`. It moves the parameters and buffers residing on a meta device
    onto ``device`` and initializes them.

    Usage:

    >>> model = MyModel(..., device=Device("meta"))
    >>>
    >>> fsdp_model = FullyShardedDataParallel(
    ...     ..., param_init_fn=FsdpParameterInitializer(Device("cuda:0"))
    ... )
    """

    _module_memo: set[Module]
    _memo: dict[Tensor, Tensor]
    _device: Device
    _skip_init: bool

    def __init__(self, device: Device, skip_init: bool = False) -> None:
        """
        :param device:
            The device onto which to move the parameters and buffers.
        :param skip_init:
            If ``True``, skips initializing the parameters and buffers after
            moving them onto ``device``. The non-persistent buffers are always
            initialized regardless of ``skip_init``.
        """
        self._module_memo = set()
        self._memo = {}
        self._device = device
        self._skip_init = skip_init

    def __call__(self, module: Module) -> None:
        if module in self._module_memo:
            return

        to_empty(module, self._device, recurse=False, memo=self._memo)

        if not self._skip_init:
            reset_parameters(module, recurse=False)
        else:
            # Non-persistent buffers are never part of module's state, so we
            # have to initialize them even with `skip_init`.
            reset_non_persistent_buffers(module, recurse=False)

        self._module_memo.add(module)


def load_with_sdp_gang(gangs: Gangs) -> ContextManager:
    try:
        pg = gangs.sdp.as_process_group()
    except NotSupportedError:
        return nullcontext()

    return load_with_process_group(pg)
