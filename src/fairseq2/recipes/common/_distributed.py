# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import final

from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.models import ModelHandler
from fairseq2.nn.data_parallel import (
    FSDP1,
    FSDP2,
    DistributedSetupError,
    FSDPGranularity,
    FSDPWrapper,
    fsdp1_load_local_state_dict,
    fsdp1_local_state_dict,
    fsdp1_summon_full_parameters,
    fsdp2_load_local_state_dict,
    fsdp2_local_state_dict,
    fsdp2_no_sync,
    fsdp2_summon_full_parameters,
    to_ddp,
    to_fsdp1,
    to_fsdp2,
)
from fairseq2.nn.utils.grad import clip_grad_norm
from fairseq2.nn.utils.module import broadcast_module, to_device
from fairseq2.recipes import Model, RecipeError
from fairseq2.recipes.config import TrainerSection
from fairseq2.typing import ContextManager

# isort: split

from fairseq2.recipes.common._error import FSDPNotSupportedError


def setup_data_parallel_model(
    context: RuntimeContext,
    trainer_section: TrainerSection,
    model: Model,
    gangs: Gangs,
    has_checkpoint: bool,
    static_graph: bool = True,
) -> Model:
    data_parallelism = trainer_section.data_parallelism

    if data_parallelism == "fsdp":
        if gangs.rdp.size > 1 and gangs.sdp.size == 1:
            log.warning("Hybrid sharded data parallelism not enabled. Falling back to DDP.")  # fmt: skip

            data_parallelism = "ddp"

    try:
        if data_parallelism == "ddp":
            return _wrap_ddp(model, gangs, static_graph)

        if data_parallelism == "fsdp":
            return _wrap_fsdp(
                trainer_section, model, gangs, has_checkpoint, static_graph
            )
    except DistributedSetupError as ex:
        raise RecipeError(
            "The data parallelism cannot be setup. See the nested exception for details."
        ) from ex

    raise ValueError("`trainer_section.data_parallelism` must be 'ddp' or 'fsdp'.")


def _wrap_ddp(model: Model, gangs: Gangs, static_graph: bool) -> Model:
    if gangs.dp.size == 1:
        to_device(model.module, gangs.root.device)

        return model

    log.info("Wrapping the model with DDP and broadcasting to all processes.")

    # We do not set DDP's `static_graph` parameter. Unfortunately, support for
    # that feature is finicky in DDP. `find_unused_parameters` is still useful
    # though and can have measurable impact on performance.
    ddp = to_ddp(model.base_module, gangs, find_unused_parameters=not static_graph)

    log.info("Model wrapped with DDP and broadcasted.")

    return _DDPModel(
        model.name, ddp, model.config, model.handler, model.newly_initialized
    )


@final
class _DDPModel(Model):
    _name: str
    _ddp: DDP
    _config: object
    _handler: ModelHandler
    _newly_initialized: bool

    def __init__(
        self,
        name: str,
        ddp: DDP,
        config: object,
        handler: ModelHandler,
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
    def module(self) -> Module:
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
    def handler(self) -> ModelHandler:
        return self._handler

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized


def _wrap_fsdp(
    trainer_section: TrainerSection,
    model: Model,
    gangs: Gangs,
    has_checkpoint: bool,
    static_graph: bool,
) -> Model:
    if not model.handler.supports_fsdp:
        raise FSDPNotSupportedError(model.name)

    if gangs.dp.size == 1:
        to_device(model.module, gangs.root.device)

        return model

    if trainer_section.mixed_precision == "static":
        mp_dtype = trainer_section.dtype
    else:
        mp_dtype = None

    def apply_fsdp(
        module: Module, granularity: FSDPGranularity, wrapper: FSDPWrapper
    ) -> None:
        model.handler.apply_fsdp(module, granularity, wrapper)

    if trainer_section.fsdp.version == "v1":
        if has_checkpoint:
            log.info("Wrapping the model with FSDP1.")  # fmt: skip
        else:
            log.info("Wrapping the model with FSDP1 and broadcasting to all processes.")  # fmt: skip

        fsdp1 = to_fsdp1(
            model.base_module,
            gangs,
            apply_fsdp,
            granularity=trainer_section.fsdp.granularity,
            mixed_precision_dtype=mp_dtype,
            fp32_reduce=trainer_section.fsdp.fp32_reduce,
            broadcast_state=not has_checkpoint,
            skip_init=has_checkpoint,
            reshard_after_forward=trainer_section.fsdp.reshard_after_forward,
        )

        if has_checkpoint:
            log.info("Model wrapped with FSDP1.")
        else:
            log.info("Model wrapped with FSDP1 and broadcasted.")

        return _FSDP1Model(
            model.name, fsdp1, model.config, model.handler, model.newly_initialized
        )
    else:
        if has_checkpoint:
            log.info("Wrapping the model with FSDP2.")  # fmt: skip
        else:
            log.info("Wrapping the model with FSDP2 and broadcasting to all processes.")  # fmt: skip

        fsdp2 = to_fsdp2(
            model.base_module,
            gangs,
            apply_fsdp,
            granularity=trainer_section.fsdp.granularity,
            mixed_precision_dtype=mp_dtype,
            fp32_reduce=trainer_section.fsdp.fp32_reduce,
            broadcast_state=not has_checkpoint,
            skip_init=has_checkpoint,
            reshard_after_forward=trainer_section.fsdp.reshard_after_forward,
        )

        if has_checkpoint:
            log.info("Model wrapped with FSDP2.")
        else:
            log.info("Model wrapped with FSDP2 and broadcasted.")

        return _FSDP2Model(
            model.name, fsdp2, model.config, model.handler, model.newly_initialized
        )


@final
class _FSDP1Model(Model):
    _name: str
    _fsdp1: FSDP1
    _config: object
    _handler: ModelHandler
    _newly_initialized: bool

    def __init__(
        self,
        name: str,
        fsdp1: FSDP1,
        config: object,
        handler: ModelHandler,
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
    def module(self) -> Module:
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
    def handler(self) -> ModelHandler:
        return self._handler

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized


@final
class _FSDP2Model(Model):
    _name: str
    _fsdp2: FSDP2
    _config: object
    _handler: ModelHandler
    _newly_initialized: bool

    def __init__(
        self,
        name: str,
        fsdp2: FSDP2,
        config: object,
        handler: ModelHandler,
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
    def module(self) -> Module:
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
    def handler(self) -> ModelHandler:
        return self._handler

    @property
    @override
    def newly_initialized(self) -> bool:
        return self._newly_initialized


def broadcast_model(model: Model, gangs: Gangs) -> None:
    if gangs.dp.size == 1:
        return

    log.info("Broadcasting '{}' model to all processes.", model.name)

    try:
        broadcast_module(model.module, gangs.dp)
    except GangError as ex:
        raise RecipeError(
            f"The '{model.name}' model cannot be broadcasted from rank 0 to the rest of the data parallel gang. See the nested exception for details."
        ) from ex

    log.info("Model broadcasted.")
