# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Literal, Protocol, cast, final, runtime_checkable

from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDPModule
from typing_extensions import override

from fairseq2.checkpoint import CheckpointManager
from fairseq2.data_type import DataType
from fairseq2.error import InternalError
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.models import ModelFamily
from fairseq2.nn.fsdp import (
    FSDP1Module,
    FSDP2Module,
    FSDPApplier,
    FSDPWrapper,
    fsdp1_load_local_state_dict,
    fsdp1_local_state_dict,
    fsdp1_summon_full_parameters,
    fsdp2_load_local_state_dict,
    fsdp2_local_state_dict,
    fsdp2_no_sync,
    fsdp2_summon_full_parameters,
)
from fairseq2.nn.utils.grad import clip_grad_norm
from fairseq2.nn.utils.module import load_state_dict, to_device
from fairseq2.recipe.config import TrainerSection
from fairseq2.recipe.error import FSDPNotSupportedError
from fairseq2.recipe.model import RecipeModel
from fairseq2.runtime.lookup import Lookup
from fairseq2.typing import ContextManager


class _DPModelWrapper(ABC):
    @abstractmethod
    def wrap(self, model: RecipeModel) -> RecipeModel: ...


@final
class _DelegatingDPModelWrapper(_DPModelWrapper):
    def __init__(
        self,
        section: TrainerSection,
        gangs: Gangs,
        dp_wrappers: Lookup[_DPModelWrapper],
    ) -> None:
        self._section = section
        self._gangs = gangs
        self._dp_wrappers = dp_wrappers

    @override
    def wrap(self, model: RecipeModel) -> RecipeModel:
        if self._gangs.dp.size == 1:
            to_device(model.module, self._gangs.root.device)

            return model

        data_parallelism = self._section.data_parallelism

        if data_parallelism == "fsdp":
            if self._gangs.rdp.size > 1 and self._gangs.sdp.size == 1:
                log.warning("Hybrid sharded data parallelism not enabled. Falling back to DDP.")  # fmt: skip

                data_parallelism = "ddp"

        wrapper = self._dp_wrappers.maybe_get(data_parallelism)
        if wrapper is None:
            raise InternalError(f"`section.data_parallelism` is '{data_parallelism}'.")

        return wrapper.wrap(model)


@runtime_checkable
class _DDPFactory(Protocol):
    def __call__(
        self, module: Module, gangs: Gangs, *, find_unused_parameters: bool
    ) -> DDPModule: ...


@final
class _DDPModelWrapper(_DPModelWrapper):
    def __init__(
        self, ddp_factory: _DDPFactory, gangs: Gangs, static_graph: bool
    ) -> None:
        self._ddp_factory = ddp_factory
        self._gangs = gangs
        self._static_graph = static_graph

    @override
    def wrap(self, model: RecipeModel) -> RecipeModel:
        log.info("Wrapping model with DDP and broadcasting to all processes.")

        # We do not set DDP's `static_graph` parameter. Unfortunately, support for
        # that feature is finicky in DDP. `find_unused_parameters` is still useful
        # though and can have measurable impact on performance.
        try:
            module = self._ddp_factory(
                model.base_module,
                self._gangs,
                find_unused_parameters=not self._static_graph,
            )
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Model wrapped with DDP and broadcasted.")

        return _DDPModel(module, model.config, model.family, model.newly_initialized)


@final
class _DDPModel(RecipeModel):
    def __init__(
        self,
        ddp_module: DDPModule,
        config: object,
        family: ModelFamily,
        newly_initialized: bool,
    ) -> None:
        self._ddp_module = ddp_module
        self._config = config
        self._family = family
        self._newly_initialized = newly_initialized

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
    def family(self) -> ModelFamily:
        return self._family

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized


@runtime_checkable
class _FSDPFactory(Protocol):
    def __call__(
        self,
        module: Module,
        gangs: Gangs,
        applier: FSDPApplier,
        *,
        version: Literal["v1", "v2"],
        mixed_precision_dtype: DataType | None,
        fp32_reduce: bool,
        broadcast_state: bool,
        skip_init: bool,
        reshard_after_forward: bool,
    ) -> Module: ...


@final
class _FSDPModelWrapper(_DPModelWrapper):
    def __init__(
        self,
        fsdp_factory: _FSDPFactory,
        section: TrainerSection,
        checkpoint_manager: CheckpointManager,
        gangs: Gangs,
    ) -> None:
        self._fsdp_factory = fsdp_factory
        self._section = section
        self._checkpoint_manager = checkpoint_manager
        self._gangs = gangs

    @override
    def wrap(self, model: RecipeModel) -> RecipeModel:
        if not model.family.supports_fsdp:
            raise FSDPNotSupportedError()

        fsdp_config = self._section.fsdp

        def apply_fsdp(module: Module, wrapper: FSDPWrapper) -> Module:
            if fsdp_config.granularity == "model":
                return wrapper(module, reshard_after_forward=False)

            return model.family.apply_fsdp(module, fsdp_config.granularity, wrapper)

        if self._section.mixed_precision.mode == "static":
            mp_dtype = self._section.mixed_precision.dtype
        else:
            mp_dtype = None

        has_checkpoint = self._checkpoint_manager.has_checkpoint()

        if self._section.fsdp.version == "v1":
            if has_checkpoint:
                log.info("Wrapping model with FSDP1.")  # fmt: skip
            else:
                log.info("Wrapping model with FSDP1 and broadcasting to all processes.")  # fmt: skip

            try:
                module = self._fsdp_factory(
                    model.base_module,
                    self._gangs,
                    apply_fsdp,
                    version="v1",
                    mixed_precision_dtype=mp_dtype,
                    fp32_reduce=fsdp_config.fp32_reduce,
                    broadcast_state=not has_checkpoint,
                    skip_init=has_checkpoint,
                    reshard_after_forward=fsdp_config.reshard_after_forward,
                )
            except GangError as ex:
                raise_operational_gang_error(ex)

            fsdp1 = cast(FSDP1Module, module)

            if has_checkpoint:
                log.info("Model wrapped with FSDP1.")
            else:
                log.info("Model wrapped with FSDP1 and broadcasted.")

            return _FSDP1Model(
                fsdp1, model.config, model.family, model.newly_initialized
            )
        else:
            if has_checkpoint:
                log.info("Wrapping model with FSDP2.")  # fmt: skip
            else:
                log.info("Wrapping model with FSDP2 and broadcasting to all processes.")  # fmt: skip

            try:
                module = self._fsdp_factory(
                    model.base_module,
                    self._gangs,
                    apply_fsdp,
                    version="v2",
                    mixed_precision_dtype=mp_dtype,
                    fp32_reduce=fsdp_config.fp32_reduce,
                    broadcast_state=not has_checkpoint,
                    skip_init=has_checkpoint,
                    reshard_after_forward=fsdp_config.reshard_after_forward,
                )
            except GangError as ex:
                raise_operational_gang_error(ex)

            fsdp2 = cast(FSDP2Module, module)

            if has_checkpoint:
                log.info("Model wrapped with FSDP2.")
            else:
                log.info("Model wrapped with FSDP2 and broadcasted.")

            return _FSDP2Model(
                fsdp2, model.config, model.family, model.newly_initialized
            )


@final
class _FSDP1Model(RecipeModel):
    def __init__(
        self,
        fsdp1_module: FSDP1Module,
        config: object,
        family: ModelFamily,
        newly_initialized: bool,
    ) -> None:
        self._fsdp1_module = fsdp1_module
        self._config = config
        self._family = family
        self._newly_initialized = newly_initialized

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
    def family(self) -> ModelFamily:
        return self._family

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized


@final
class _FSDP2Model(RecipeModel):
    def __init__(
        self,
        fsdp2_module: FSDP2Module,
        config: object,
        family: ModelFamily,
        newly_initialized: bool,
    ) -> None:
        self._fsdp2_module = fsdp2_module
        self._config = config
        self._family = family
        self._newly_initialized = newly_initialized

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
    def family(self) -> ModelFamily:
        return self._family

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized
