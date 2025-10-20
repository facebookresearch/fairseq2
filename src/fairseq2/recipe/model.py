# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, TypeVar, final

from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDPModule
from typing_extensions import override

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
from fairseq2.recipe.error import ModelTypeNotValidError
from fairseq2.typing import ContextManager, Stateful

ModelT = TypeVar("ModelT", bound=Module)


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

    def base_as(self, kls: type[ModelT]) -> ModelT:
        if not isinstance(self.base_module, kls):
            raise ModelTypeNotValidError(type(self.base_module), kls, self.section_name)

        return self.base_module

    def check_base_type(self, kls: type[ModelT]) -> None:
        self.base_as(kls)

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
    def family_name(self) -> str: ...

    @property
    @abstractmethod
    def newly_initialized(self) -> bool: ...

    @property
    @abstractmethod
    def section_name(self) -> str: ...


@final
class StandardRecipeModel(RecipeModel):
    def __init__(
        self,
        module: Module,
        config: object,
        family_name: str,
        *,
        newly_initialized: bool = False,
        section_name: str = "model",
    ) -> None:
        self._module = module
        self._config = config
        self._family_name = family_name
        self._newly_initialized = newly_initialized
        self._section_name = section_name

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
    def family_name(self) -> str:
        return self._family_name

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized

    @property
    @override
    def section_name(self) -> str:
        return self._section_name


@final
class DDPModel(RecipeModel):
    def __init__(
        self,
        ddp_module: DDPModule,
        config: object,
        family_name: str,
        *,
        newly_initialized: bool = False,
        section_name: str = "model",
    ) -> None:
        self._ddp_module = ddp_module
        self._config = config
        self._family_name = family_name
        self._newly_initialized = newly_initialized
        self._section_name = section_name

    @override
    def state_dict(self) -> dict[str, object]:
        return self._ddp_module.module.state_dict()  # type: ignore[no-any-return]

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        load_state_dict(self._ddp_module.module, state_dict)

    @override
    def no_sync(self) -> ContextManager[None]:
        return self._ddp_module.no_sync()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._ddp_module, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager[None]:
        return nullcontext()

    @property
    @override
    def module(self) -> Module:
        return self._ddp_module

    @property
    @override
    def base_module(self) -> Module:
        return self._ddp_module.module  # type: ignore[no-any-return]

    @property
    @override
    def config(self) -> object:
        return self._config

    @property
    @override
    def family_name(self) -> str:
        return self._family_name

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized

    @property
    @override
    def section_name(self) -> str:
        return self._section_name


@final
class FSDP1Model(RecipeModel):
    def __init__(
        self,
        fsdp1_module: FSDP1Module,
        config: object,
        family_name: str,
        *,
        newly_initialized: bool = False,
        section_name: str = "model",
    ) -> None:
        self._fsdp1_module = fsdp1_module
        self._config = config
        self._family_name = family_name
        self._newly_initialized = newly_initialized
        self._section_name = section_name

    @override
    def state_dict(self) -> dict[str, object]:
        return fsdp1_local_state_dict(self._fsdp1_module)

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        fsdp1_load_local_state_dict(self._fsdp1_module, state_dict)

    @override
    def no_sync(self) -> ContextManager[None]:
        return self._fsdp1_module.no_sync()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._fsdp1_module, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager[None]:
        return fsdp1_summon_full_parameters(self._fsdp1_module)

    @property
    @override
    def module(self) -> Module:
        return self._fsdp1_module

    @property
    @override
    def base_module(self) -> Module:
        return self._fsdp1_module.module

    @property
    @override
    def config(self) -> object:
        return self._config

    @property
    @override
    def family_name(self) -> str:
        return self._family_name

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized

    @property
    @override
    def section_name(self) -> str:
        return self._section_name


@final
class FSDP2Model(RecipeModel):
    def __init__(
        self,
        fsdp2_module: FSDP2Module,
        config: object,
        family_name: str,
        *,
        newly_initialized: bool = False,
        section_name: str = "model",
    ) -> None:
        self._fsdp2_module = fsdp2_module
        self._config = config
        self._family_name = family_name
        self._newly_initialized = newly_initialized
        self._section_name = section_name

    @override
    def state_dict(self) -> dict[str, object]:
        return fsdp2_local_state_dict(self._fsdp2_module)

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        fsdp2_load_local_state_dict(self._fsdp2_module, state_dict)

    @override
    def no_sync(self) -> ContextManager[None]:
        return fsdp2_no_sync(self._fsdp2_module)

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._fsdp2_module, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager[None]:
        return fsdp2_summon_full_parameters(self._fsdp2_module)

    @property
    @override
    def module(self) -> Module:
        return self._fsdp2_module

    @property
    @override
    def base_module(self) -> Module:
        return self._fsdp2_module

    @property
    @override
    def config(self) -> object:
        return self._config

    @property
    @override
    def family_name(self) -> str:
        return self._family_name

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized

    @property
    @override
    def section_name(self) -> str:
        return self._section_name
