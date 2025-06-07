# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.models.handler import ModelFamilyHandler
from fairseq2.nn.utils.grad import clip_grad_norm
from fairseq2.typing import ContextManager


class Model(ABC):
    @abstractmethod
    def state_dict(self) -> dict[str, object]: ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, object]) -> None: ...

    @abstractmethod
    def no_sync(self) -> ContextManager: ...

    @abstractmethod
    def clip_grad_norm(self, max_norm: float | None) -> Tensor: ...

    @abstractmethod
    def summon_full_parameters(self) -> ContextManager: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

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
    def handler(self) -> ModelFamilyHandler: ...

    @property
    @abstractmethod
    def newly_initialized(self) -> bool: ...


@final
class StandardModel(Model):
    def __init__(
        self,
        name: str,
        model: Module,
        config: object,
        handler: ModelFamilyHandler,
        newly_initialized: bool = False,
    ) -> None:
        self._name = name
        self._model = model
        self._config = config
        self._handler = handler
        self._newly_initialized = newly_initialized

    @override
    def state_dict(self) -> dict[str, object]:
        return self._model.state_dict()

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self._model.load_state_dict(state_dict)

    @override
    def no_sync(self) -> ContextManager:
        return nullcontext()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._model, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager:
        return nullcontext()

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def module(self) -> Module:
        return self._model

    @property
    @override
    def base_module(self) -> Module:
        return self._model

    @property
    @override
    def config(self) -> object:
        return self._config

    @property
    @override
    def handler(self) -> ModelFamilyHandler:
        return self._handler

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized
