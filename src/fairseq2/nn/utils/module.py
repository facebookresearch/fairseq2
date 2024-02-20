# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from dataclasses import dataclass
from logging import Logger
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    final,
    runtime_checkable,
)

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from fairseq2.typing import CPU, META, Device


@runtime_checkable
class ModuleWithParameter(Protocol):
    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""


def reset_parameters(module: Module, *, recurse: bool = True) -> None:
    """Reset the parameters and buffers of ``module``.

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
    """Reset the non-persistent buffers of ``module``.

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


def load_state_dict(module: Module, state_dict: Mapping[str, Any]) -> None:
    """Copy parameters and buffers from ``state_dict`` into ``module`` and its
    descendants.

    This implementation internally calls :meth:`Module.load_state_dict()` with
    ``strict`` set to ``True``, and also enforces that ``state_dict`` does not
    contain any keys corresponding to descendants that are set to ``None`` via
    :meth:`Module.register_module()`.
    """
    module.load_state_dict(state_dict, strict=True)

    unexpected_keys = []

    for name, descendant in _named_modules(module):
        if descendant is not None:
            continue

        prefix = name + "."

        for key in state_dict.keys():
            if key.startswith(prefix):
                unexpected_keys.append(key)

    if unexpected_keys:
        raise RuntimeError(
            f"Unexpected key(s) in `state_dict`: {', '.join(unexpected_keys)}"
        )


def _named_modules(
    module: Optional[Module], memo: Optional[Set[Module]] = None, prefix: str = ""
) -> Iterator[Tuple[str, Optional[Module]]]:
    if module is None:
        yield prefix, None

        return

    if memo is None:
        memo = set()
    elif module in memo:
        return

    memo.add(module)

    yield prefix, module

    # This loop is the main reason why this function is internal. There is no
    # formal way to retrieve the list of all descendants via PyTorch APIs. The
    # only workaround is to use the internal `_modules` attribute.
    for name, descendant in module._modules.items():
        if not prefix:
            descendant_prefix = name
        else:
            descendant_prefix = prefix + "." + name

        yield from _named_modules(descendant, memo, descendant_prefix)


def select_parameters(
    module: Module, names: Sequence[str], *, exclude: bool = False
) -> Iterable[Tuple[str, Parameter]]:
    """Select the parameters of ``module`` and its descendants whose name
    matches ``names``.

    :param module:
        The module to check.
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


def to_device(module: Module, device: Device) -> None:
    """Move the parameters and buffers of ``module`` to ``device``.

    :param module:
        The module to move.
    :param device:
        The target device of the parameters and buffers.
    """
    module_device = infer_device(module)
    if module_device == device:
        return

    if module_device != META:
        module.to(device=device)
    else:
        to_empty(module, device=device)

        reset_parameters(module)


def apply_depth_first(
    module: Module, fn: Callable[[Module], None], memo: Optional[Set[Module]] = None
) -> None:
    """Apply ``fn`` to ``module`` and it descendants in depth-first order.

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
    for child in module.children():
        if child is not None:
            apply_depth_first(child, fn, memo)

    fn(module)


def freeze(module: Module, value: bool) -> None:
    """Set if ``module`` and its descendants should stop learning (i.e. freeze)."""
    for param in module.parameters():
        param.requires_grad_(not value)


def infer_device(module: Module, param_name: Optional[str] = None) -> Device:
    """Infer the device on which ``module``'s parameters and buffers reside."""
    devices = set()

    for param in module.parameters():
        devices.add(param.device)

    for buf in module.buffers():
        devices.add(buf.device)

    if len(devices) == 0:
        return CPU

    if len(devices) == 1:
        return devices.pop()

    if param_name is None:
        param_name = "module"

    s = ", ".join(sorted(f"'{d.type}'" for d in devices))

    raise ValueError(
        f"All parameters and buffers of `{param_name}` must be on the same device, but they are on {s}."
    )


@final
@dataclass
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


def get_module_size(module: Module) -> ModuleSizeInfo:
    """Return the size information of ``module`` and its descendants."""
    info = ModuleSizeInfo()

    for param in module.parameters():
        if param is not None:
            size = param.numel()
            size_bytes = size * param.element_size()

            info.param_size += size
            info.param_size_bytes += size_bytes

            if param.requires_grad:
                info.trainable_param_size += size
                info.trainable_param_size_bytes += size_bytes

            info.total_size += size
            info.total_size_bytes += size_bytes

    for buf in module.buffers():
        size = buf.numel()
        size_bytes = size * param.element_size()

        info.buffer_size += size
        info.buffer_size_bytes += size * size_bytes

        info.total_size += size
        info.total_size_bytes += size_bytes

    return info


def log_module(module: Module, logger: Logger) -> None:
    """Log information about ``module`` and its descendants."""
    if not logger.isEnabledFor(logging.INFO):
        return

    info = []

    size_info = get_module_size(module)

    info.append(f"Parameter Size: {size_info.param_size:,}")
    info.append(f"Parameter Size (bytes): {size_info.param_size_bytes:,}")
    info.append(f"Trainable Parameter Size: {size_info.trainable_param_size:,}")
    info.append(f"Trainable Parameter Size (bytes): {size_info.trainable_param_size_bytes:,}")  # fmt: skip
    info.append(f"Buffer Size: {size_info.buffer_size:,}")
    info.append(f"Buffer Size (bytes): {size_info.buffer_size_bytes:,}")
    info.append(f"Total Size: {size_info.total_size:,}")
    info.append(f"Total Size (bytes): {size_info.total_size_bytes:,}")

    s = " | ".join(info)

    logger.info(f"Module - {s}\n{module}")
