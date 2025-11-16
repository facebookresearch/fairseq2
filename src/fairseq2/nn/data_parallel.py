# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Final, cast, final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.error import InternalError
from fairseq2.nn.ddp import DDPModule
from fairseq2.nn.fsdp import (
    FSDP1Module,
    FSDP2Module,
    fsdp1_load_local_state_dict,
    fsdp1_local_state_dict,
    fsdp1_summon_full_parameters,
    fsdp2_load_local_state_dict,
    fsdp2_local_state_dict,
    fsdp2_no_sync,
    fsdp2_summon_full_parameters,
)
from fairseq2.nn.utils.grad import clip_grad_norm
from fairseq2.nn.utils.module import load_state_dict
from fairseq2.typing import ContextManager, Stateful


class DataParallelFacade(ABC, Stateful):
    """
    Provides an API-agnostic way to interact with different data parallelism
    implementations.

    DDP, FSDP, and other data parallelism implementations expose different APIs
    for operations such as state handling and gradient clipping. This interface
    acts as a facade, providing a consistent way to access these underlying APIs.
    """

    @abstractmethod
    def state_dict(self) -> dict[str, object]: ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, object]) -> None: ...

    @abstractmethod
    def no_sync(self) -> ContextManager[None]: ...

    @abstractmethod
    def clip_grad_norm(self, max_norm: float | None) -> Tensor: ...

    @abstractmethod
    def summon_full_parameters(self) -> ContextManager[None]: ...


@final
class _NoopDataParallelFacade(DataParallelFacade):
    def __init__(self, module: Module) -> None:
        self._module = module

    @override
    def state_dict(self) -> dict[str, object]:
        return self._module.state_dict()

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        load_state_dict(self._module, state_dict)

    @override
    def no_sync(self) -> ContextManager[None]:
        return nullcontext()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._module, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager[None]:
        return nullcontext()


@final
class _DDPFacade(DataParallelFacade):
    def __init__(self, module: DDPModule) -> None:
        self._weak_module = weakref.ref(module)

    @override
    def state_dict(self) -> dict[str, object]:
        module = self._get_module()

        return module.module.state_dict()  # type: ignore[no-any-return]

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        module = self._get_module()

        load_state_dict(module.module, state_dict)

    @override
    def no_sync(self) -> ContextManager[None]:
        module = self._get_module()

        return module.no_sync()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        module = self._get_module()

        return clip_grad_norm(module, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager[None]:
        return nullcontext()

    def _get_module(self) -> DDPModule:
        module = self._weak_module()
        if module is None:
            raise InternalError("`module` has already been deallocated.")

        return module


@final
class _FSDP1Facade(DataParallelFacade):
    def __init__(self, module: FSDP1Module) -> None:
        self._weak_module = weakref.ref(module)

    @override
    def state_dict(self) -> dict[str, object]:
        module = self._get_module()

        return fsdp1_local_state_dict(module)

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        module = self._get_module()

        fsdp1_load_local_state_dict(module, state_dict)

    @override
    def no_sync(self) -> ContextManager[None]:
        module = self._get_module()

        return module.no_sync()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        module = self._get_module()

        return clip_grad_norm(module, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager[None]:
        module = self._get_module()

        return fsdp1_summon_full_parameters(module)

    def _get_module(self) -> FSDP1Module:
        module = self._weak_module()
        if module is None:
            raise InternalError("`module` has already been deallocated.")

        return module


@final
class _FSDP2Facade(DataParallelFacade):
    def __init__(self, module: FSDP2Module) -> None:
        self._weak_module = weakref.ref(module)

    @override
    def state_dict(self) -> dict[str, object]:
        module = self._get_module()

        return fsdp2_local_state_dict(module)

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        module = self._get_module()

        fsdp2_load_local_state_dict(module, state_dict)

    @override
    def no_sync(self) -> ContextManager[None]:
        module = self._get_module()

        return fsdp2_no_sync(module)

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        module = self._get_module()

        return clip_grad_norm(module, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager[None]:
        module = self._get_module()

        return fsdp2_summon_full_parameters(module)

    def _get_module(self) -> FSDP2Module:
        module = self._weak_module()
        if module is None:
            raise InternalError("`module` has already been deallocated.")

        return module


_FACADE_KEY: Final = "__fs2_dp_facade__"


def set_data_parallel_facade(module: Module, facade: DataParallelFacade) -> None:
    """
    Associates ``facade`` with the specified module.
    """
    setattr(module, _FACADE_KEY, facade)


def get_data_parallel_facade(module: Module) -> DataParallelFacade:
    """
    Returns the data parallel facade associated with the specified module.

    If ``module`` is of type :class:`DDPModule`, :class:`FSDP1Module`, or :class:`FSDP2Module`,
    this function will return the corresponding facade, even if one was not
    previously set.

    If the module is not a data parallel module and has no facade, this function
    will return a no-op implementation.
    """
    facade = getattr(module, _FACADE_KEY, None)

    if facade is not None:
        if not isinstance(facade, DataParallelFacade):
            raise InternalError(f"{_FACADE_KEY} is of type `{type(facade)}`.")

        return facade

    if isinstance(module, DDPModule):
        return _DDPFacade(module)

    if isinstance(module, FSDP1Module):
        return _FSDP1Facade(module)

    if isinstance(module, FSDP2Module):
        return _FSDP2Facade(module)

    return _NoopDataParallelFacade(module)
