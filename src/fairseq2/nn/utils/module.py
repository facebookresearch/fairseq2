# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Protocol, Set, runtime_checkable

from torch.nn import Module

from fairseq2.typing import Device


@runtime_checkable
class ModuleWithParameter(Protocol):
    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""


def reset_parameters(module: Module) -> None:
    """Reset the parameters and buffers of ``module`` and its submodules.

    :param module:
        The module to reset.
    """

    def maybe_reset(module: Module) -> None:
        if isinstance(module, ModuleWithParameter):
            module.reset_parameters()

    apply_depth_first(module, maybe_reset)


@runtime_checkable
class ModuleWithNonPersistentBuffer(Protocol):
    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""


def reset_non_persistent_buffers(module: Module) -> None:
    """Reset the non-persistent buffers of ``module`` and its submodules.

    :param module:
        The module to reset.
    """

    def maybe_reset(module: Module) -> None:
        if isinstance(module, ModuleWithNonPersistentBuffer):
            module.reset_non_persistent_buffers()

    apply_depth_first(module, maybe_reset)


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


def freeze(module: Module, value: bool) -> None:
    """Change if ``module`` and its submodules should freeze (i.e. stop learning)."""
    for param in module.parameters():
        param.requires_grad_(not value)


def infer_device(module: Module) -> Device:
    """Infer the device on which ``module``'s parameter(s) reside."""
    try:
        param = next(iter(module.parameters()))
    except StopIteration:
        return Device("cpu")

    return param.device
