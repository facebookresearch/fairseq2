# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP

from fairseq2.context import RuntimeContext
from fairseq2.error import NotSupportedError, ProgramError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.models.fsdp import get_fsdp_wrap_policy
from fairseq2.nn.ddp import DistributedSetupError, to_ddp
from fairseq2.nn.fsdp import to_fsdp
from fairseq2.nn.utils.module import broadcast_module, to_device
from fairseq2.recipes.config import TrainerSection, get_config_section
from fairseq2.utils.env import InvalidEnvironmentVariableError, get_local_world_size


def setup_data_parallel_model(
    context: RuntimeContext,
    recipe_config: object,
    base_model: Module,
    gangs: Gangs,
    static_graph: bool = True,
) -> Module:
    trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)

    data_parallelism = trainer_section.data_parallelism

    try:
        if trainer_section.fsdp.hsdp:
            try:
                local_world_size = get_local_world_size(os.environ)
            except InvalidEnvironmentVariableError as ex:
                raise DistributedSetupError(
                    "The local world size for HSDP cannot be determined. See the nested exception for details."
                ) from ex

            if local_world_size == 1:
                data_parallelism = "ddp"

                log.warning("`trainer.fsdp.hsdp` is set, but the local world size is 1. Falling back to DDP.")  # fmt: skip
        else:
            local_world_size = None

        if data_parallelism == "ddp":
            return wrap_ddp(base_model, gangs, static_graph)

        if data_parallelism == "fsdp":
            return wrap_fsdp(
                trainer_section, base_model, gangs, static_graph, local_world_size
            )
    except DistributedSetupError as ex:
        raise ProgramError(
            "The data parallelism cannot be setup. See the nested exception for details."
        ) from ex

    raise ValueError(
        "`recipe_config.trainer.data_parallelism` must be 'ddp' or 'fsdp'."
    )


def wrap_ddp(base_model: Module, gangs: Gangs, static_graph: bool) -> Module:
    if gangs.dp.size == 1:
        to_device(base_model, gangs.root.device)

        return base_model

    log.info("Wrapping the model with DDP and broadcasting to all processes.")

    # We do not set DDP's `static_graph` parameter. Unfortunately, support for
    # that feature is finicky in DDP. `find_unused_parameters` is still useful
    # though and can have measurable impact on perfomance.
    dp_model = to_ddp(base_model, gangs.dp, find_unused_parameters=not static_graph)

    log.info("Model wrapped with DDP and broadcasted.")

    return dp_model


def wrap_fsdp(
    trainer_section: TrainerSection,
    base_model: Module,
    gangs: Gangs,
    static_graph: bool,
    local_world_size: int | None,
) -> Module:
    if trainer_section.fsdp.version == "v2":
        raise NotSupportedDistributedFeature("FSDP2 is not supported yet.")

    if not static_graph:
        raise NotSupportedDistributedFeature(
            "FSDP does not support non-static model graphs."
        )

    if local_world_size is not None:
        if gangs.root.size != gangs.dp.size:
            raise NotSupportedDistributedFeature(
                "HSDP cannot be used with non-data parallelism."
            )

    if gangs.dp.size == 1:
        to_device(base_model, gangs.root.device)

        return base_model

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
        broadcast_state=True,
        reshard_after_forward=trainer_section.fsdp.reshard_after_forward,
        local_world_size=local_world_size,
        mixed_precision_dtype=mp_dtype,
        fp32_reduce=trainer_section.fsdp.fp32_reduce,
    )

    log.info("Model wrapped with FSDP and broadcasted.")

    return dp_model


class NotSupportedDistributedFeature(NotSupportedError):
    pass


def broadcast_model(name: str, model: Module, gangs: Gangs) -> None:
    if gangs.dp.size == 1:
        return

    log.info("Broadcasting '{}' model to all processes.", name)

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
