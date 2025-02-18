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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.error import NotSupportedError, ProgramError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.models import ModelHandler
from fairseq2.models.fsdp import get_fsdp_wrap_policy
from fairseq2.nn.data_parallel import (
    DistributedSetupError,
    get_fsdp_full_state_dict,
    get_fsdp_optim_state_dict,
    load_fsdp_optim_state_dict,
    summon_fsdp,
    to_ddp,
    to_fsdp,
)
from fairseq2.nn.utils.gradient import clip_gradient_norm
from fairseq2.nn.utils.module import broadcast_module, to_device
from fairseq2.recipes.config import TrainerSection, get_config_section
from fairseq2.recipes.error import (
    HybridShardingNotSupportedError,
    StaticGraphNotSupportedError,
)
from fairseq2.recipes.model import Model
from fairseq2.typing import ContextManager


def setup_data_parallel_model(
    context: RuntimeContext,
    recipe_config: object,
    model: Model,
    gangs: Gangs,
    static_graph: bool = True,
) -> Model:
    trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)

    data_parallelism = trainer_section.data_parallelism

    if data_parallelism == "fsdp":
        if gangs.rdp.size > 1 and gangs.sdp.size == 1:
            log.warning("Hybrid sharded data parallelism not enabled. Falling back to DDP.")  # fmt: skip

            data_parallelism = "ddp"

    try:
        if data_parallelism == "ddp":
            return wrap_ddp(model, gangs, static_graph)

        if data_parallelism == "fsdp":
            return wrap_fsdp(recipe_config, model, gangs, static_graph)
    except DistributedSetupError as ex:
        raise ProgramError(
            "The data parallelism cannot be setup. See the nested exception for details."
        ) from ex

    raise ValueError(
        "`recipe_config.trainer.data_parallelism` must be 'ddp' or 'fsdp'."
    )


def wrap_ddp(model: Model, gangs: Gangs, static_graph: bool) -> Model:
    if gangs.dp.size == 1:
        to_device(model.module, gangs.root.device)

        return model

    log.info("Wrapping the model with DDP and broadcasting to all processes.")

    # We do not set DDP's `static_graph` parameter. Unfortunately, support for
    # that feature is finicky in DDP. `find_unused_parameters` is still useful
    # though and can have measurable impact on performance.
    dp_module = to_ddp(model.module, gangs, find_unused_parameters=not static_graph)

    log.info("Model wrapped with DDP and broadcasted.")

    return DDPModel(dp_module, model)


@final
class DDPModel(Model):
    _ddp: DDP
    _wrapped_model: Model

    def __init__(self, ddp: DDP, wrapped_model: Model) -> None:
        self._ddp = ddp
        self._wrapped_model = wrapped_model

    @override
    def no_sync(self) -> ContextManager:
        return self._ddp.no_sync()

    @override
    def clip_gradient_norm(self, max_norm: float | None) -> Tensor:
        return clip_gradient_norm(self._ddp, max_norm)

    @override
    def state_dict(self) -> dict[str, object]:
        return self._ddp.state_dict()

    @override
    def optim_state_dict(self, optim: Optimizer) -> dict[str, object]:
        return optim.state_dict()  # type: ignore[no-any-return]

    @override
    def load_optim_state_dict(
        self, optim: Optimizer, state_dict: Mapping[str, object]
    ) -> None:
        optim.load_state_dict(state_dict)

    @override
    def summon_parameters(self) -> ContextManager:
        return nullcontext()

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
    def name(self) -> str:
        return self._wrapped_model.name

    @property
    @override
    def config(self) -> object:
        return self._wrapped_model.config

    @property
    @override
    def handler(self) -> ModelHandler:
        return self._wrapped_model.handler

    @property
    @override
    def is_empty_init(self) -> bool:
        return self._wrapped_model.is_empty_init


def wrap_fsdp(
    recipe_config: object, model: Model, gangs: Gangs, static_graph: bool
) -> Model:
    trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)

    if trainer_section.fsdp.version == "v2":
        raise NotSupportedError("FSDP2 is not supported yet.")

    if not static_graph:
        raise StaticGraphNotSupportedError("FSDP")

    if gangs.dp.size == 1:
        to_device(model.module, gangs.root.device)

        return model

    if gangs.rdp.size > 1:
        if gangs.root.size != gangs.dp.size:  # means we have model parallelism.
            raise HybridShardingNotSupportedError("FSDP")

    log.info("Wrapping the model with FSDP and broadcasting to all processes.")  # fmt: skip

    if trainer_section.mixed_precision == "static":
        mp_dtype = trainer_section.dtype
    else:
        mp_dtype = None

    wrap_policy, ignored_modules = get_fsdp_wrap_policy(
        model.module, wrap_granularity=trainer_section.fsdp.granularity
    )

    dp_module = to_fsdp(
        model.module,
        gangs,
        wrap_policy,
        ignored_modules=ignored_modules,
        broadcast_state=True,
        reshard_after_forward=trainer_section.fsdp.reshard_after_forward,
        mixed_precision_dtype=mp_dtype,
        fp32_reduce=trainer_section.fsdp.fp32_reduce,
    )

    log.info("Model wrapped with FSDP and broadcasted.")

    return FSDPModel(dp_module, model)


@final
class FSDPModel(Model):
    _fsdp: FSDP
    _wrapped_model: Model

    def __init__(self, fsdp: FSDP, wrapped_model: Model) -> None:
        self._fsdp = fsdp
        self._wrapped_model = wrapped_model

    @override
    def no_sync(self) -> ContextManager:
        return self._fsdp.no_sync()

    @override
    def clip_gradient_norm(self, max_norm: float | None) -> Tensor:
        return clip_gradient_norm(self._fsdp, max_norm)

    @override
    def state_dict(self) -> dict[str, object]:
        return get_fsdp_full_state_dict(self._fsdp)

    @override
    def optim_state_dict(self, optim: Optimizer) -> dict[str, object]:
        return get_fsdp_optim_state_dict(self._fsdp, optim)

    @override
    def load_optim_state_dict(
        self, optim: Optimizer, state_dict: Mapping[str, object]
    ) -> None:
        load_fsdp_optim_state_dict(self._fsdp, optim, state_dict)

    @override
    def summon_parameters(self) -> ContextManager:
        return summon_fsdp(self._fsdp)

    @property
    @override
    def module(self) -> Module:
        return self._fsdp

    @property
    @override
    def base_module(self) -> Module:
        return self._fsdp.module

    @property
    @override
    def name(self) -> str:
        return self._wrapped_model.name

    @property
    @override
    def config(self) -> object:
        return self._wrapped_model.config

    @property
    @override
    def handler(self) -> ModelHandler:
        return self._wrapped_model.handler

    @property
    @override
    def is_empty_init(self) -> bool:
        return self._wrapped_model.is_empty_init


def broadcast_model(model: Model, gangs: Gangs) -> None:
    if gangs.dp.size == 1:
        return

    log.info("Broadcasting '{}' model to all processes.", model.name)

    try:
        broadcast_module(model.module, gangs.dp)
    except GangError as ex:
        raise ProgramError(
            f"The '{model.name}' model cannot be broadcasted from rank 0 to the rest of the gang. See the nested exception for details."
        ) from ex

    log.info("Model broadcasted.")
