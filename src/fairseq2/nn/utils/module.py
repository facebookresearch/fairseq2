# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional, Protocol, Set, runtime_checkable

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from fairseq2.typing import CPU, Device


@runtime_checkable
class ModuleWithParameter(Protocol):
    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""


def reset_parameters(module: Module, *, recurse: bool = True) -> None:
    """Reset the parameters and buffers of ``module`` and its submodules.

    :param module:
        The module to reset.
    :param recurse:
        If ``True``, resets the parameters of all descendant modules.
    """

    def maybe_reset(m: Module) -> None:
        if isinstance(m, ModuleWithParameter):
            m.reset_parameters()

    if recurse:
        apply_depth_first(module, maybe_reset)
    else:
        maybe_reset(module)


@runtime_checkable
class ModuleWithNonPersistentBuffer(Protocol):
    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""


def reset_non_persistent_buffers(module: Module, *, recurse: bool = True) -> None:
    """Reset the non-persistent buffers of ``module`` and its submodules.

    :param module:
        The module to reset.
    :param recurse:
        If ``True``, resets the non-persistent buffers of all descendant modules.
    """

    def maybe_reset(m: Module) -> None:
        if isinstance(m, ModuleWithNonPersistentBuffer):
            m.reset_non_persistent_buffers()

    if recurse:
        apply_depth_first(module, maybe_reset)
    else:
        maybe_reset(module)


def to_empty(
    module: Module,
    device: Device,
    *,
    recurse: bool = True,
    memo: Optional[Dict[Tensor, Tensor]] = None,
) -> None:
    """Move the parameters and buffers of ``module`` to ``device`` without
    copying storage.

    :param module:
        The module to move.
    :param device:
        The target device of the parameters and buffers.
    :param recurse:
        If ``True``, moves the parameters and buffers of all descendant modules.
    :param memo:
        The memoization dictionary to use to detect shared parameters and
        buffers. If ``None``, constructs an internal one.
    """
    if memo is None:
        memo = {}

    def empty_like(src: Tensor) -> Tensor:
        if src in memo:
            return memo[src]

        tgt = torch.empty_like(src, device=device)

        memo[src] = tgt

        return tgt

    def to_empty_(m: Module) -> None:
        for name, prm in m.named_parameters(recurse=False):
            if prm is None:
                continue

            with torch.no_grad():
                new_prm = Parameter(empty_like(prm), prm.requires_grad)

            setattr(m, name, new_prm)

            if (grad := prm.grad) is not None:
                with torch.no_grad():
                    new_grad = empty_like(grad).requires_grad_(grad.requires_grad)

                new_prm.grad = new_grad

        for name, buf in m.named_buffers(recurse=False):
            if buf is None:
                continue

            setattr(m, name, empty_like(buf))

    if recurse:
        apply_depth_first(module, to_empty_)
    else:
        to_empty_(module)


def apply_depth_first(
    module: Module, fn: Callable[[Module], None], memo: Optional[Set[Module]] = None
) -> None:
    """Apply ``fn`` to ``module`` and it submodules in a depth-first order.

    :param module:
        The module to process.
    :param fn:
        The function to apply to ``module``.
    :param memo:
        The module container to use for memoization.
    """
    if memo is None:
        memo = set()
    elif module in memo:
        return

    # Do not apply more than once.
    memo.add(module)

    # Depth first so that the children are handled first.
    for submodule in module.children():
        if submodule is not None:
            apply_depth_first(submodule, fn, memo)

    fn(module)


class FSDPParameterInitializer:
    """Initializes the parameters and buffers of an FSDP module.

    This is a convenience callable to pass to the ``param_init_fn`` parameter of
    the FSDP constructor. It moves the parameters and buffers residing on a meta
    device to ``device`` and initializes them.

    Usage:

    >>> model = MyModel(..., device=Device("meta"))
    >>>
    >>> fsdp_model = FullyShardedDataParallel(
    ...     ..., param_init_fn=FSDPParameterInitializer(Device("cuda:0"))
    ... )
    """

    memo: Dict[Tensor, Tensor]
    device: Device

    def __init__(self, device: Device) -> None:
        """
        :param device:
            The device on which to initialize the parameters and buffers.
        """
        self.memo = {}
        self.device = device

    def __call__(self, module: Module) -> None:
        """
        :param module:
            An FSDP module or submodule.
        """
        to_empty(module, self.device, recurse=False, memo=self.memo)

        reset_parameters(module, recurse=False)


def freeze(module: Module, value: bool) -> None:
    """Change if ``module`` and its submodules should freeze (i.e. stop learning)."""
    for param in module.parameters():
        param.requires_grad_(not value)


def infer_device(module: Module) -> Device:
    """Infer the device on which ``module``'s parameter(s) reside."""
    try:
        param = next(iter(module.parameters()))
    except StopIteration:
        return CPU

    return param.device
