# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import (
    Callable,
    Dict,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

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


def select_parameters(
    module: Module, names: Sequence[str], *, exclude: bool = False
) -> Iterable[Tuple[str, Parameter]]:
    """Select the parameters of ``module`` matching ``names``.

    :param module:
        The module to select from.
    :param names:
        The parameter names. Can contain regular expressions.
    :param exclude:
        If ``True``, return the parameters that do not match ``names``.

    :returns:
        An iterable of name-parameter tuples.
    """
    for name, param in module.named_parameters():
        matched = any(name == pattern or re.match(pattern, name) for pattern in names)

        if (matched and not exclude) or (not matched and exclude):
            yield name, param


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
        for name, param in m.named_parameters(recurse=False):
            if param is None:
                continue

            with torch.no_grad():
                new_param = Parameter(empty_like(param), param.requires_grad)

            setattr(m, name, new_param)

            if (grad := param.grad) is not None:
                with torch.no_grad():
                    new_grad = empty_like(grad).requires_grad_(grad.requires_grad)

                new_param.grad = new_grad

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
    """Apply ``fn`` to ``module`` and it submodules in depth-first order.

    :param module:
        The module to process.
    :param fn:
        The function to apply to ``module``.
    :param memo:
        The memoization set to use to detect visited modules. If ``None``,
        constructs an internal one.
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
    """Set if ``module`` and its submodules should stop learning (i.e. freeze)."""
    for param in module.parameters():
        param.requires_grad_(not value)


def infer_device(module: Module) -> Device:
    """Infer the device on which ``module``'s parameters reside."""
    try:
        param = next(iter(module.parameters()))
    except StopIteration:
        return CPU

    return param.device
