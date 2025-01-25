# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP

from fairseq2.checkpoint import CheckpointManager
from fairseq2.context import RuntimeContext
from fairseq2.error import NotSupportedError, ProgramError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.models.fsdp import get_fsdp_wrap_policy
from fairseq2.nn.ddp import to_ddp
from fairseq2.nn.fsdp import to_fsdp
from fairseq2.nn.utils.module import broadcast_module, to_device
from fairseq2.recipes.config import TrainerSection, get_config_section


def setup_data_parallel_model(
    context: RuntimeContext,
    recipe_config: object,
    base_model: Module,
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
    static_graph: bool = True,
) -> Module:
    trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)

    if trainer_section.data_parallelism == "ddp":
        ddp_wrapper = DdpWrapper()

        return ddp_wrapper.wrap(base_model, gangs, static_graph)

    if trainer_section.data_parallelism == "fsdp":
        fsdp_wrapper = FsdpWrapper()

        return fsdp_wrapper.wrap(
            trainer_section, base_model, gangs, checkpoint_manager, static_graph
        )

    raise ValueError(
        "`recipe_config.trainer.data_parallelism` must be 'ddp' or 'fsdp'."
    )


@final
class DdpWrapper:
    def wrap(self, base_model: Module, gangs: Gangs, static_graph: bool) -> Module:
        if gangs.dp.size == 1:
            to_device(base_model, gangs.root.device)

            return base_model

        log.info("Wrapping the model with DDP and broadcasting to all processes.")

        dp_model = to_ddp(
            base_model,
            gangs.dp,
            find_unused_parameters=not static_graph,
            static_graph=static_graph,
        )

        log.info("Model wrapped with DDP and broadcasted.")

        return dp_model


@final
class FsdpWrapper:
    def wrap(
        self,
        trainer_section: TrainerSection,
        base_model: Module,
        gangs: Gangs,
        checkpoint_manager: CheckpointManager,
        static_graph: bool,
    ) -> Module:
        if gangs.dp.size == 1:
            to_device(base_model, gangs.root.device)

            return base_model

        if not static_graph:
            raise NotSupportedError("FSDP does not support non-static graphs.")

        has_checkpoint = checkpoint_manager.has_checkpoint()

        if has_checkpoint:
            log.info("Wrapping the model with FSDP.")
        else:
            log.info("Wrapping the model with FSDP and broadcasting to all processes.")  # fmt: skip

        if trainer_section.mixed_precision == "static":
            mp_dtype = trainer_section.dtype
        else:
            mp_dtype = None

        wrap_policy, ignored_modules = get_fsdp_wrap_policy(
            base_model, wrap_granularity=trainer_section.fsdp.granularity
        )

        dp_model = to_fsdp(
            base_model,
            gangs.dp,
            wrap_policy,
            ignored_modules=ignored_modules,
            broadcast_state=not has_checkpoint,
            reshard_after_forward=trainer_section.fsdp.reshard_after_forward,
            mixed_precision_dtype=mp_dtype,
            fp32_reduce=trainer_section.fsdp.fp32_reduce,
        )

        if has_checkpoint:
            log.info("Model wrapped with FSDP.")
        else:
            log.info("Model wrapped with FSDP and broadcasted.")

        return dp_model


def broadcast_model(name: str, model: Module, gangs: Gangs) -> None:
    if gangs.dp.size == 1:
        return

    log.info("Broadcasting the '{}' model to all processes.", name)

    try:
        broadcast_module(model, gangs.dp)
    except GangError as ex:
        raise ProgramError(
            "The '{}' model cannot be broadcasted from rank 0 to the rest of the gang. See the nested exception for details."
        ) from ex

    log.info("Model broadcasted.")


def check_model_type(model: Module, kls: type[Module]) -> None:
    """Check if a potentially DDP or FSDP wrapped `model` is of type `kls`."""
    if isinstance(model, (DDP, FSDP)):
        model = model.module

    if not isinstance(model, kls):
        raise TypeError(
            f"`model` must be of type `{kls}`, but is of type `{type(model)}` instead."
        )
