# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module

from fairseq2.error import InternalError
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.model.context import (
    DDPModelContext,
    FSDP1ModelContext,
    FSDP2ModelContext,
    ModelContext,
)
from fairseq2.nn.data_parallel import (
    FSDPGranularity,
    FSDPWrapper,
    to_ddp,
    to_fsdp1,
    to_fsdp2,
)
from fairseq2.nn.utils.module import to_device
from fairseq2.recipe.checkpoint import _check_has_checkpoint
from fairseq2.recipe.config import TrainerSection, get_config_section
from fairseq2.recipe.error import FSDPNotSupportedError
from fairseq2.runtime.dependency import DependencyResolver


def _create_data_parallel_model(
    resolver: DependencyResolver, model_context: ModelContext
) -> ModelContext:
    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    gangs = resolver.resolve(Gangs)

    data_parallelism = trainer_section.data_parallelism

    if data_parallelism == "fsdp":
        if gangs.rdp.size > 1 and gangs.sdp.size == 1:
            log.warning("Hybrid sharded data parallelism not enabled. Falling back to DDP.")  # fmt: skip

            data_parallelism = "ddp"

    if data_parallelism == "ddp":
        return _wrap_ddp(resolver, model_context)

    if data_parallelism == "fsdp":
        return _wrap_fsdp(resolver, model_context)

    raise InternalError(f"`trainer_section.data_parallelism` is '{data_parallelism}'.")


def _wrap_ddp(
    resolver: DependencyResolver, model_context: ModelContext
) -> ModelContext:
    static_graph = resolver.resolve(bool, key="static_graph")

    gangs = resolver.resolve(Gangs)

    if gangs.dp.size == 1:
        to_device(model_context.model, gangs.root.device)

        return model_context

    log.info("Wrapping the model with DDP and broadcasting to all processes.")

    # We do not set DDP's `static_graph` parameter. Unfortunately, support for
    # that feature is finicky in DDP. `find_unused_parameters` is still useful
    # though and can have measurable impact on performance.
    ddp = to_ddp(
        model_context.base_module, gangs, find_unused_parameters=not static_graph
    )

    log.info("Model wrapped with DDP and broadcasted.")

    return DDPModelContext(
        model_context.name,
        ddp,
        model_context.config,
        model_context.handler,
        model_context.newly_initialized,
    )


def _wrap_fsdp(
    resolver: DependencyResolver, model_context: ModelContext
) -> ModelContext:
    gangs = resolver.resolve(Gangs)

    if not model_context.handler.supports_fsdp:
        raise FSDPNotSupportedError(model_context.name)

    if gangs.dp.size == 1:
        to_device(model_context.model, gangs.root.device)

        return model_context

    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    if trainer_section.mixed_precision == "static":
        mp_dtype = trainer_section.dtype
    else:
        mp_dtype = None

    has_checkpoint = _check_has_checkpoint(resolver)

    def apply_fsdp(
        module: Module, granularity: FSDPGranularity, wrapper: FSDPWrapper
    ) -> None:
        model_context.handler.apply_fsdp(module, granularity, wrapper)

    if trainer_section.fsdp.version == "v1":
        if has_checkpoint:
            log.info("Wrapping the model with FSDP1.")  # fmt: skip
        else:
            log.info("Wrapping the model with FSDP1 and broadcasting to all processes.")  # fmt: skip

        fsdp1 = to_fsdp1(
            model_context.base_module,
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

        return FSDP1ModelContext(
            model_context.name,
            fsdp1,
            model_context.config,
            model_context.handler,
            model_context.newly_initialized,
        )
    else:
        if has_checkpoint:
            log.info("Wrapping the model with FSDP2.")  # fmt: skip
        else:
            log.info("Wrapping the model with FSDP2 and broadcasting to all processes.")  # fmt: skip

        fsdp2 = to_fsdp2(
            model_context.base_module,
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

        return FSDP2ModelContext(
            model_context.name,
            fsdp2,
            model_context.config,
            model_context.handler,
            model_context.newly_initialized,
        )
