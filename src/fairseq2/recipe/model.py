# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.models import ModelFamily
from fairseq2.nn.data_parallel import get_data_parallel_facade
from fairseq2.recipe.internal.model import _ModelHolder
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
    def __init__(self, model_holder: _ModelHolder) -> None:
        dp_facade = get_data_parallel_facade(model_holder.dp_model)

        self._model_holder = model_holder
        self._dp_facade = dp_facade

    @override
    def state_dict(self) -> dict[str, object]:
        return self._dp_facade.state_dict()

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        return self._dp_facade.load_state_dict(state_dict)

    @override
    def no_sync(self) -> ContextManager[None]:
        return self._dp_facade.no_sync()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return self._dp_facade.clip_grad_norm(max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager[None]:
        return self._dp_facade.summon_full_parameters()

    @property
    @override
    def module(self) -> Module:
        return self._model_holder.dp_model

    @property
    @override
    def base_module(self) -> Module:
        return self._model_holder.model

    @property
    @override
    def config(self) -> object:
        return self._model_holder.config

    @property
    @override
    def family(self) -> ModelFamily:
        return self._model_holder.family

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._model_holder.newly_initialized
