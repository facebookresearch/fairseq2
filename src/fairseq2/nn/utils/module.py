# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Protocol, runtime_checkable

import torch

from fairseq2.gang import Gang
from fairseq2.logging import log
from fairseq2.typing import CPU, Device
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.utils import remove_weight_norm  # type: ignore[attr-defined]


@runtime_checkable
class ModuleWithParameter(Protocol):
    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""


def reset_parameters(module: Module, *, recurse: bool = True) -> None:
    """Reset the parameters and buffers of ``module``.

    :param module:
        The module to reset.
    :param recurse:
        If ``True``, resets the parameters and buffers of descendant modules.
    """

    def reset(name: str, m: Module) -> None:
        if isinstance(m, ModuleWithParameter):
            m.reset_parameters()

    visit_module(module, reset, recurse=recurse)


@runtime_checkable
class ModuleWithNonPersistentBuffer(Protocol):
    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""


def reset_non_persistent_buffers(module: Module, *, recurse: bool = True) -> None:
    """Reset the non-persistent buffers of ``module``.

    :param module:
        The module to reset.
    :param recurse:
        If ``True``, resets the non-persistent buffers of descendant modules.
    """

    def reset(name: str, m: Module) -> None:
        if isinstance(m, ModuleWithNonPersistentBuffer):
            m.reset_non_persistent_buffers()

    visit_module(module, reset, recurse=recurse)


def visit_module(
    module: Module,
    visitor: Callable[[str, Module], None],
    *,
    recurse: bool = True,
    post_order: bool = True,
    memo: set[Module] | None = None,
) -> None:
    """Run ``visitor`` on ``module``.

    :param module:
        The module to visit.
    :param visitor:
        The visitor to run on ``module``.
    :param recurse:
        If ``True``, visits descendant modules.
    :param post_order:
        If ``True``, visits descendant modules first.
    :param memo:
        The memoization set to detect visited modules. If ``None``, constructs
        an internal one.
    """
    for name, m in _get_named_modules(
        module, recurse=recurse, post_order=post_order, memo=memo
    ):
        if m is not None:
            visitor(name, m)


def to_device(module: Module, device: Device) -> None:
    """Move the parameters and buffers of ``module`` to ``device``.

    :param module:
        The module to move.
    :param device:
        The target device of the parameters and buffers.
    """
    modules: list[tuple[Module, Device]] = []

    for name, m in _get_named_modules(module, prefix="module", post_order=True):
        if m is None:
            continue

        try:
            module_device = infer_device(m, recurse=False)
        except ValueError as ex:
            raise ValueError(
                f"The device of `{name}` is not valid. See the nested exception for details."
            ) from ex

        if module_device == device:
            continue

        modules.append((m, module_device))

    if not modules:
        return

    memo: dict[Tensor, Tensor] = {}

    for m, module_device in modules:
        if module_device.type != "meta":
            apply_to_parameters(m, lambda t: t.to(device), recurse=False, memo=memo)
        else:
            to_empty(m, device, recurse=False, memo=memo)

            reset_parameters(m, recurse=False)


def to_empty(
    module: Module,
    device: Device,
    *,
    recurse: bool = True,
    memo: dict[Tensor, Tensor] | None = None,
) -> None:
    """Move the parameters and buffers of ``module`` to ``device`` without
    copying storage.

    :param module:
        The module to move.
    :param device:
        The target device.
    :param recurse:
        If ``True``, moves the parameters and buffers of descendant modules.
    :param memo:
        The memoization dictionary to detect shared parameters and buffers. If
        ``None``, constructs an internal one.
    """

    def convert(source: Tensor) -> Tensor:
        return torch.empty_like(source, device=device)

    apply_to_parameters(module, convert, recurse=recurse, memo=memo)


def share_parameters(source_module: Module, target_module: Module) -> None:
    """Share the parameters and buffers of ``source_module`` with ``target_module``.

    :param source_module:
        The module whose parameters and buffers will be shared.
    :param target_module:
        The module whose parameters and buffers will be overwritten.
    """
    sources = chain(source_module.named_parameters(), source_module.named_buffers())
    targets = chain(target_module.named_parameters(), target_module.named_buffers())

    for (src_name, src_tensor), (tgt_name, tgt_tensor) in zip(sources, targets):
        if src_name != tgt_name:
            raise ValueError(
                f"`source_module` and `target_module` must have matching parameters and buffers, but `target_module` has no '{src_name}'."
            )

        if src_tensor.grad is not None:
            raise ValueError(
                f"The parameters must not have their `grad` set, but '{src_name}' of `source_module` has it set."
            )

        if tgt_tensor.grad is not None:
            raise ValueError(
                f"The parameters must not have their `grad` set, but '{tgt_name}' of `target_module` has it set."
            )

    tensors = []

    # The order of the collected tensors here must match `apply_to_parameters()`.
    def collect_tensors(m: Module) -> None:
        for child in m.children():
            if child is not None:
                collect_tensors(child)

        for tensor in chain(m.parameters(recurse=False), m.buffers(recurse=False)):
            if tensor is not None:
                tensors.append(tensor)

    collect_tensors(source_module)

    if not tensors:
        return

    it = iter(tensors)

    # Do not memoize. No need anyways, and would also break the sync between the
    # traversed tensors and the iterator.
    apply_to_parameters(target_module, lambda _: next(it), no_memo=True)


def apply_to_parameters(
    module: Module,
    fn: Callable[[Tensor], Tensor],
    *,
    recurse: bool = True,
    memo: dict[Tensor, Tensor] | None = None,
    no_memo: bool = False,
) -> None:
    """Apply ``fn`` to the parameters and buffers of ``module``.

    :param module:
        The module to process.
    :param fn:
        The function to apply.
    :param recurse:
        If ``True``, applies ``fn`` to the parameters and buffers of descendant
        modules.
    :param memo:
        The memoization dictionary to detect shared parameters and buffers. If
        ``None`` and ``no_memo`` is ``False``, constructs an internal one.
    :param no_memo:
        If ``True``, skips memoization.
    """
    if no_memo:
        memo = None
    elif memo is None and recurse:
        memo = {}

    if recurse:
        for child in module.children():
            if child is not None:
                apply_to_parameters(
                    child, fn, recurse=recurse, memo=memo, no_memo=no_memo
                )

    def call_fn(
        source: Tensor, is_param: bool = False, requires_grad: bool = False
    ) -> Tensor:
        if memo is not None and source in memo:
            return memo[source]

        target = fn(source)

        if is_param:
            target = Parameter(target, requires_grad)
        elif requires_grad:
            target.requires_grad_(requires_grad)

        if memo is not None:
            memo[source] = target

        return target

    for param_name, param in module.named_parameters(recurse=False):
        if param is None:
            continue

        with torch.no_grad():
            new_param = call_fn(param, is_param=True, requires_grad=param.requires_grad)

        setattr(module, param_name, new_param)

        if (grad := param.grad) is not None:
            with torch.no_grad():
                new_grad = call_fn(grad, requires_grad=grad.requires_grad)

            new_param.grad = new_grad

    for buffer_name, buffer in module.named_buffers(recurse=False):
        if buffer is None:
            continue

        setattr(module, buffer_name, call_fn(buffer))


def freeze_parameters(module: Module | None, value: bool = True) -> None:
    """Set if ``module`` and its descendant modules should stop learning."""
    if module is None:
        return

    module.requires_grad_(not value)


def select_parameters(
    module: Module, names: Sequence[str], *, exclude: bool = False
) -> Iterable[tuple[str, Parameter]]:
    """Select the parameters of ``module`` and its descendant modules whose
    names match ``names``.

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


def remove_parametrizations(module: Module, *, recurse: bool = True) -> None:
    """Remove parametrizations on ``module`` (e.g. :func:`~torch.nn.utils.weight_norm`).

    :param module:
        The module to process.
    :param recurse:
        If ``True``, removes parametrizations on descendant modules.
    """

    def remove(name: str, m: Module) -> None:
        try:
            remove_weight_norm(m)

            setattr(m, "no_parametrization", True)
        except ValueError:
            pass

    visit_module(module, remove, recurse=recurse)


def infer_device(module: Module, *, recurse: bool = True) -> Device:
    """Infer the device on which ``module``'s parameters and buffers reside.

    :param module:
        The module to check.
    :param name:
        The name of the module for error reporting purposes.
    :param recurse:
        If ``True``, infers the device by checking the parameters and buffers of
        descendant modules as well.
    """
    devices = set()

    for param in module.parameters(recurse):
        devices.add(param.device)

    for buffer in module.buffers(recurse):
        devices.add(buffer.device)

    if len(devices) == 0:
        return CPU

    if len(devices) == 1:
        return devices.pop()

    s = ", ".join(sorted(f"`{d.type}`" for d in devices))

    raise ValueError(
        f"All parameters and buffers of `module` must be on the same device, but they are on {s}."
    )


def broadcast_module(
    module: Module,
    gang: Gang,
    *,
    source_rank: int = 0,
    non_persistent_buffers: bool = False,
    skip_modules: set[Module] | None = None,
) -> None:
    """Broadcasts ``module`` to all processes in ``gang``.

    :param module: The module to broadcast.
    :param gang The gang over which to broadcast ``module``.
    :param source_rank: The rank of the source process from which to broadcast.
    :param non_persistent_buffers: If ``True``, broadcasts the non-persistent
        buffers as well.
    :param skip_modules: The set of modules that won't be broadcasted.
    """
    to_device(module, gang.device)

    if gang.size == 1:
        return

    if skip_modules is None:
        skip_modules = set()

    warned = False

    memo: set[Tensor] = set()

    tensors = []

    def collect_tensors(m: Module) -> None:
        nonlocal warned

        if m in skip_modules:
            return

        for child in m.children():
            collect_tensors(child)

        for param in m.parameters(recurse=False):
            if param in memo:
                continue

            memo.add(param)

            tensors.append(param.detach())

            if not warned and param.grad is not None:
                log.warning("`broadcast_module()` does not support syncing gradients, but one or more parameters of `module` have their `grads` defined.")  # fmt: skip

                warned = True

        for buffer_name, buffer in m.named_buffers(recurse=False):
            if buffer in memo:
                continue

            memo.add(buffer)

            if not non_persistent_buffers:
                # TODO(balioglu): Surprisingly, PyTorch still does not offer a
                # public API to check the type of a module buffer. This should
                # be updated in the future.
                if buffer_name in m._non_persistent_buffers_set:
                    continue

            tensors.append(buffer.detach())

    collect_tensors(module)

    if not tensors:
        return

    pg = gang.as_process_group()

    bucket_size = 250 * 1024 * 1024  # Same as DDP bucket size.

    from torch.distributed import _broadcast_coalesced

    # TODO(balioglu): Call c10d in fairseq2n instead.
    _broadcast_coalesced(pg, tensors, bucket_size, source_rank)


def load_state_dict(
    module: Module, state_dict: Mapping[str, object], strict: bool = True
) -> None:
    """Copy parameters and buffers from ``state_dict`` into ``module`` and its
    descendant modules.

    This implementation internally calls :meth:`Module.load_state_dict()`, and also enforces that
    ``state_dict`` does not contain any keys corresponding to descendants that are set to ``None``
    via :meth:`Module.register_module()`.
    """
    # Key mapping
    need_mapping = False
    sample_key = list(state_dict.keys())[0]
    if (
        sample_key.startswith("module.")
        and not sample_key in module.state_dict().keys()
    ):
        mapped_key = sample_key[7:]
        if mapped_key in module.state_dict().keys():
            need_mapping = True

    if need_mapping:
        key_mapping = lambda key: key[7:] if key.startswith("module.") else key
        state_dict = {key_mapping(key): value for key, value in state_dict.items()}

    module.load_state_dict(state_dict, strict=strict)

    unexpected_keys = []

    for name, m in _get_named_modules(module):
        if m is not None:
            continue

        prefix = name + "."

        for key in state_dict.keys():
            if key.startswith(prefix):
                unexpected_keys.append(key)

    if unexpected_keys:
        unexpected_keys.sort()

        s = ", ".join(unexpected_keys)

        raise ValueError(
            f"`state_dict` must not contain the following unexpected key(s): {s}"
        )


def _get_named_modules(
    module: Module | None,
    *,
    prefix: str = "",
    recurse: bool = True,
    post_order: bool = False,
    memo: set[Module] | None = None,
) -> Iterator[tuple[str, Module | None]]:
    if module is None:
        yield prefix, None

        return

    if memo is None and recurse:
        memo = set()

    if memo is not None:
        if module in memo:
            return

        memo.add(module)

    if not post_order:
        yield prefix, module

    if recurse:
        # This loop is the reason why this function is internal. There is no
        # PyTorch API to retrieve the list of all descendant modules. We use
        # the internal `_modules` attribute as a workaround.
        for descendant_name, descendant in module._modules.items():
            if not prefix:
                descendant_prefix = descendant_name
            else:
                descendant_prefix = prefix + "." + descendant_name

            yield from _get_named_modules(
                descendant,
                prefix=descendant_prefix,
                recurse=recurse,
                post_order=post_order,
                memo=memo,
            )

    if post_order:
        yield prefix, module


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


def get_module_size(module: Module) -> ModuleSizeInfo:
    """Return the size information of ``module`` and its descendant modules."""
    info = ModuleSizeInfo()

    for param in module.parameters():
        if param is None:
            continue

        size = param.numel()
        size_bytes = size * param.element_size()

        info.param_size += size
        info.param_size_bytes += size_bytes

        if param.requires_grad:
            info.trainable_param_size += size
            info.trainable_param_size_bytes += size_bytes

        info.total_size += size
        info.total_size_bytes += size_bytes

    for buffer in module.buffers():
        if buffer is None:
            continue

        size = buffer.numel()
        size_bytes = size * buffer.element_size()

        info.buffer_size += size
        info.buffer_size_bytes += size_bytes

        info.total_size += size
        info.total_size_bytes += size_bytes

    return info
