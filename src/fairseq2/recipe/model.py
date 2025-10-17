# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.models import ModelFamily
from fairseq2.nn.utils.grad import clip_grad_norm
from fairseq2.nn.utils.module import load_state_dict
from fairseq2.typing import ContextManager, Stateful


class RecipeModel(ABC, Stateful):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

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

    @property
    @abstractmethod
    def module(self) -> Module: ...

    @property
    @abstractmethod
    def base_module(self) -> Module: ...

    @property
    @abstractmethod
    def config(self) -> object: ...

    @property
    @abstractmethod
    def family(self) -> ModelFamily: ...

    @property
    @abstractmethod
    def newly_initialized(self) -> bool: ...


@final
class _StandardRecipeModel(RecipeModel):
    def __init__(
        self,
        module: Module,
        config: object,
        family: ModelFamily,
        newly_initialized: bool = False,
    ) -> None:
        self._module = module
        self._config = config
        self._family = family
        self._newly_initialized = newly_initialized

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

    @property
    @override
    def module(self) -> Module:
        return self._module

    @property
    @override
    def base_module(self) -> Module:
        return self._module

    @property
    @override
    def config(self) -> object:
        return self._config

    @property
    @override
    def family(self) -> ModelFamily:
        return self._family

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized
