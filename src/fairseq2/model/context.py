# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import nullcontext
from typing import final

from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import override

from fairseq2.models.handler import ModelFamilyHandler
from fairseq2.nn.data_parallel import (
    FSDP1,
    FSDP2,
    fsdp1_load_local_state_dict,
    fsdp1_local_state_dict,
    fsdp1_summon_full_parameters,
    fsdp2_load_local_state_dict,
    fsdp2_local_state_dict,
    fsdp2_no_sync,
    fsdp2_summon_full_parameters,
)
from fairseq2.nn.utils.grad import clip_grad_norm
from fairseq2.typing import ContextManager


class ModelContext(ABC):
    @abstractmethod
    def state_dict(self) -> dict[str, object]: ...

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None: ...

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
    def model(self) -> Module: ...

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
class StandardModelContext(ModelContext):
    _name: str
    _model: Module
    _config: object
    _handler: ModelFamilyHandler
    _newly_initialized: bool

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
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
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
    def model(self) -> Module:
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


@final
class DDPModelContext(ModelContext):
    _name: str
    _ddp: DDP
    _config: object
    _handler: ModelFamilyHandler
    _newly_initialized: bool

    def __init__(
        self,
        name: str,
        ddp: DDP,
        config: object,
        handler: ModelFamilyHandler,
        newly_initialized: bool,
    ) -> None:
        self._name = name
        self._ddp = ddp
        self._config = config
        self._handler = handler
        self._newly_initialized = newly_initialized

    @override
    def state_dict(self) -> dict[str, object]:
        return self._ddp.module.state_dict()  # type: ignore[no-any-return]

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self._ddp.module.load_state_dict(state_dict)

    @override
    def no_sync(self) -> ContextManager:
        return self._ddp.no_sync()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._ddp, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager:
        return nullcontext()

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def model(self) -> Module:
        return self._ddp

    @property
    @override
    def base_module(self) -> Module:
        return self._ddp.module  # type: ignore[no-any-return]

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


@final
class FSDP1ModelContext(ModelContext):
    _name: str
    _fsdp1: FSDP1
    _config: object
    _handler: ModelFamilyHandler
    _newly_initialized: bool

    def __init__(
        self,
        name: str,
        fsdp1: FSDP1,
        config: object,
        handler: ModelFamilyHandler,
        newly_initialized: bool,
    ) -> None:
        self._name = name
        self._fsdp1 = fsdp1
        self._config = config
        self._handler = handler
        self._newly_initialized = newly_initialized

    @override
    def state_dict(self) -> dict[str, object]:
        return fsdp1_local_state_dict(self._fsdp1)

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        fsdp1_load_local_state_dict(self._fsdp1, state_dict)

    @override
    def no_sync(self) -> ContextManager:
        return self._fsdp1.no_sync()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._fsdp1, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager:
        return fsdp1_summon_full_parameters(self._fsdp1)

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def model(self) -> Module:
        return self._fsdp1

    @property
    @override
    def base_module(self) -> Module:
        return self._fsdp1.module

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


@final
class FSDP2ModelContext(ModelContext):
    _name: str
    _fsdp2: FSDP2
    _config: object
    _handler: ModelFamilyHandler
    _newly_initialized: bool

    def __init__(
        self,
        name: str,
        fsdp2: FSDP2,
        config: object,
        handler: ModelFamilyHandler,
        newly_initialized: bool,
    ) -> None:
        self._name = name
        self._fsdp2 = fsdp2
        self._config = config
        self._handler = handler
        self._newly_initialized = newly_initialized

    @override
    def state_dict(self) -> dict[str, object]:
        return fsdp2_local_state_dict(self._fsdp2)

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        fsdp2_load_local_state_dict(self._fsdp2, state_dict)

    @override
    def no_sync(self) -> ContextManager:
        return fsdp2_no_sync(self._fsdp2)

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._fsdp2, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager:
        return fsdp2_summon_full_parameters(self._fsdp2)

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def model(self) -> Module:
        return self._fsdp2

    @property
    @override
    def base_module(self) -> Module:
        return self._fsdp2

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
