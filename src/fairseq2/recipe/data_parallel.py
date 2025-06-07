# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Literal, Protocol, cast, final

from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDPModule
from typing_extensions import override

from fairseq2.checkpoint import CheckpointManager
from fairseq2.data_type import DataType
from fairseq2.error import InternalError
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.model import Model
from fairseq2.models import ModelFamilyHandler
from fairseq2.nn.data_parallel import (
    FSDP1Module,
    FSDP2Module,
    FSDPApplier,
    FSDPGranularity,
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
from fairseq2.nn.utils.module import to_device
from fairseq2.recipe.config import TrainerSection
from fairseq2.runtime.provider import Provider
from fairseq2.typing import ContextManager


class DPModelWrapper(ABC):
    @abstractmethod
    def wrap(self, model: Model) -> Model: ...


@final
class DelegatingDPModelWrapper(DPModelWrapper):
    def __init__(
        self, wrappers: Provider[DPModelWrapper], section: TrainerSection, gangs: Gangs
    ) -> None:
        self._wrappers = wrappers
        self._section = section
        self._gangs = gangs

    @override
    def wrap(self, model: Model) -> Model:
        data_parallelism = self._section.data_parallelism

        if data_parallelism == "fsdp":
            if self._gangs.rdp.size > 1 and self._gangs.sdp.size == 1:
                log.warning("Hybrid sharded data parallelism not enabled. Falling back to DDP.")  # fmt: skip

                data_parallelism = "ddp"

        wrapper = self._wrappers.maybe_get(data_parallelism)
        if wrapper is None:
            raise InternalError(f"`section.data_parallelism` is '{data_parallelism}'.")

        return wrapper.wrap(model)


@final
class DDPModelWrapper(DPModelWrapper):
    def __init__(
        self, ddp_factory: DDPFactory, gangs: Gangs, static_graph: bool
    ) -> None:
        self._ddp_factory = ddp_factory
        self._gangs = gangs
        self._static_graph = static_graph

    @override
    def wrap(self, model: Model) -> Model:
        if self._gangs.dp.size == 1:
            to_device(model.module, self._gangs.root.device)

            return model

        log.info("Wrapping the model with DDP and broadcasting to all processes.")

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

        return DDPModel(
            model.name, module, model.config, model.handler, model.newly_initialized
        )


class DDPFactory(Protocol):
    def __call__(
        self, module: Module, gangs: Gangs, *, find_unused_parameters: bool
    ) -> DDPModule: ...


@final
class DDPModel(Model):
    def __init__(
        self,
        name: str,
        ddp_module: DDPModule,
        config: object,
        handler: ModelFamilyHandler,
        newly_initialized: bool,
    ) -> None:
        self._name = name
        self._ddp_module = ddp_module
        self._config = config
        self._handler = handler
        self._newly_initialized = newly_initialized

    @override
    def state_dict(self) -> dict[str, object]:
        return self._ddp_module.module.state_dict()  # type: ignore[no-any-return]

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self._ddp_module.module.load_state_dict(state_dict)

    @override
    def no_sync(self) -> ContextManager:
        return self._ddp_module.no_sync()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._ddp_module, max_norm)

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
    def handler(self) -> ModelFamilyHandler:
        return self._handler

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized


@final
class FSDPModelWrapper(DPModelWrapper):
    def __init__(
        self,
        fsdp_factory: FSDPFactory,
        section: TrainerSection,
        checkpoint_manager: CheckpointManager,
        gangs: Gangs,
    ) -> None:
        self._fsdp_factory = fsdp_factory
        self._section = section
        self._checkpoint_manager = checkpoint_manager
        self._gangs = gangs

    @override
    def wrap(self, model: Model) -> Model:
        if not model.handler.supports_fsdp:
            raise FSDPNotSupportedError(model.name)

        if self._gangs.dp.size == 1:
            to_device(model.module, self._gangs.root.device)

            return model

        fsdp_section = self._section.fsdp

        if self._section.mixed_precision == "static":
            mp_dtype = self._section.dtype
        else:
            mp_dtype = None

        has_checkpoint = self._checkpoint_manager.has_checkpoint()

        def apply_fsdp(
            module: Module, granularity: FSDPGranularity, wrapper: FSDPWrapper
        ) -> None:
            model.handler.apply_fsdp(module, granularity, wrapper)

        if self._section.fsdp.version == "v1":
            if has_checkpoint:
                log.info("Wrapping the model with FSDP1.")  # fmt: skip
            else:
                log.info("Wrapping the model with FSDP1 and broadcasting to all processes.")  # fmt: skip

            try:
                module = self._fsdp_factory(
                    model.base_module,
                    self._gangs,
                    apply_fsdp,
                    version="v1",
                    granularity=fsdp_section.granularity,
                    mixed_precision_dtype=mp_dtype,
                    fp32_reduce=fsdp_section.fp32_reduce,
                    broadcast_state=not has_checkpoint,
                    skip_init=has_checkpoint,
                    reshard_after_forward=fsdp_section.reshard_after_forward,
                )
            except GangError as ex:
                raise_operational_gang_error(ex)

            fsdp1 = cast(FSDP1Module, module)

            if has_checkpoint:
                log.info("Model wrapped with FSDP1.")
            else:
                log.info("Model wrapped with FSDP1 and broadcasted.")

            return FSDP1Model(
                model.name, fsdp1, model.config, model.handler, model.newly_initialized
            )
        else:
            if has_checkpoint:
                log.info("Wrapping the model with FSDP2.")  # fmt: skip
            else:
                log.info("Wrapping the model with FSDP2 and broadcasting to all processes.")  # fmt: skip

            try:
                module = self._fsdp_factory(
                    model.base_module,
                    self._gangs,
                    apply_fsdp,
                    version="v2",
                    granularity=fsdp_section.granularity,
                    mixed_precision_dtype=mp_dtype,
                    fp32_reduce=fsdp_section.fp32_reduce,
                    broadcast_state=not has_checkpoint,
                    skip_init=has_checkpoint,
                    reshard_after_forward=fsdp_section.reshard_after_forward,
                )
            except GangError as ex:
                raise_operational_gang_error(ex)

            fsdp2 = cast(FSDP2Module, module)

            if has_checkpoint:
                log.info("Model wrapped with FSDP2.")
            else:
                log.info("Model wrapped with FSDP2 and broadcasted.")

            return FSDP2Model(
                model.name, fsdp2, model.config, model.handler, model.newly_initialized
            )


class FSDPFactory(Protocol):
    def __call__(
        self,
        module: Module,
        gangs: Gangs,
        applier: FSDPApplier,
        *,
        version: Literal["v1", "v2"],
        granularity: FSDPGranularity,
        mixed_precision_dtype: DataType | None,
        fp32_reduce: bool,
        broadcast_state: bool,
        skip_init: bool,
        reshard_after_forward: bool,
    ) -> Module: ...


@final
class FSDP1Model(Model):
    def __init__(
        self,
        name: str,
        fsdp1_module: FSDP1Module,
        config: object,
        handler: ModelFamilyHandler,
        newly_initialized: bool,
    ) -> None:
        self._name = name
        self._fsdp1_module = fsdp1_module
        self._config = config
        self._handler = handler
        self._newly_initialized = newly_initialized

    @override
    def state_dict(self) -> dict[str, object]:
        return fsdp1_local_state_dict(self._fsdp1_module)

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        fsdp1_load_local_state_dict(self._fsdp1_module, state_dict)

    @override
    def no_sync(self) -> ContextManager:
        return self._fsdp1_module.no_sync()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._fsdp1_module, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager:
        return fsdp1_summon_full_parameters(self._fsdp1_module)

    @property
    @override
    def name(self) -> str:
        return self._name

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
    def handler(self) -> ModelFamilyHandler:
        return self._handler

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized


@final
class FSDP2Model(Model):
    def __init__(
        self,
        name: str,
        fsdp2_module: FSDP2Module,
        config: object,
        handler: ModelFamilyHandler,
        newly_initialized: bool,
    ) -> None:
        self._name = name
        self._fsdp2_module = fsdp2_module
        self._config = config
        self._handler = handler
        self._newly_initialized = newly_initialized

    @override
    def state_dict(self) -> dict[str, object]:
        return fsdp2_local_state_dict(self._fsdp2_module)

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        fsdp2_load_local_state_dict(self._fsdp2_module, state_dict)

    @override
    def no_sync(self) -> ContextManager:
        return fsdp2_no_sync(self._fsdp2_module)

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return clip_grad_norm(self._fsdp2_module, max_norm)

    @override
    def summon_full_parameters(self) -> ContextManager:
        return fsdp2_summon_full_parameters(self._fsdp2_module)

    @property
    @override
    def name(self) -> str:
        return self._name

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
    def handler(self) -> ModelFamilyHandler:
        return self._handler

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized


class FSDPNotSupportedError(Exception):
    def __init__(self, model_name: str) -> None:
        super().__init__(f"{model_name} model does not support FSDP.")

        self.model_name = model_name
