# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from datetime import timedelta

from fairseq2.context import RuntimeContext
from fairseq2.gang import (
    GangError,
    Gangs,
    setup_fsdp_gangs,
    setup_parallel_gangs,
    setup_root_gang,
)
from fairseq2.logging import log
from fairseq2.recipes import RecipeError
from fairseq2.recipes.common import HybridShardingNotSupportedError
from fairseq2.recipes.config import (
    GangSection,
    TrainerSection,
)
from fairseq2.recipes.utils.log import log_environment_info, log_ranks
from fairseq2.utils.device import DeviceDetectionError, determine_default_device
from fairseq2.utils.env import InvalidEnvironmentVariableError, get_local_world_size


def setup_gangs(context: RuntimeContext, gang_section: GangSection) -> Gangs:
    gangs = _do_setup_gangs(context, gang_section)

    log_ranks(log, gangs)

    gangs.root.barrier()

    return gangs


def setup_training_gangs(
    context: RuntimeContext, gang_section: GangSection, trainer_section: TrainerSection
) -> Gangs:
    gangs = _do_setup_gangs(context, gang_section)

    try:
        gangs = _maybe_setup_fsdp_gangs(context, trainer_section, gangs)
    except GangError as ex:
        raise RecipeError(
            "The hybrid sharded data parallel gangs cannot set up. See the nested exception for details."
        ) from ex

    log_ranks(log, gangs)

    return gangs


def _do_setup_gangs(context: RuntimeContext, gang_section: GangSection) -> Gangs:
    try:
        device = determine_default_device(context)
    except DeviceDetectionError as ex:
        raise RecipeError(
            "The device of the process cannot be determined. See the nested exception for details."
        ) from ex

    log.info("Setting '{}' as the default device of the process.", device)

    log_environment_info(log, device)

    log.info("Initializing the root gang.")

    timeout = timedelta(minutes=gang_section.timeout)

    try:
        root_gang = setup_root_gang(
            device, timeout=timeout, high_priority=gang_section.high_priority
        )
    except GangError as ex:
        raise RecipeError(
            "The root gang of the process cannot be set up. See the nested exception for details."
        ) from ex

    log.info("Root gang initialized.")

    log.info("Initializing parallel gangs.")

    try:
        tp_size = gang_section.tensor_parallel_size

        if tp_size > root_gang.size:
            raise GangError(
                f"The tensor parallel size ({tp_size}) must be less than or equal to the number of processes in the root gang."
            )

        if root_gang.size % tp_size != 0:
            raise GangError(
                f"The number of processes in the root gang is expected to be a multiple of the tensor parallel size ({tp_size}), but is {root_gang.size} instead."
            )

        gangs = setup_parallel_gangs(root_gang, tp_size=tp_size)
    except GangError as ex:
        raise RecipeError(
            "The parallel gangs of the process cannot be set up. See the nested exception for details."
        ) from ex

    log.info("Parallel gangs initialized.")

    # Perform a warm-up collective call to ensure that the internal machinery of
    # the gangs is fully set up.
    log.info("Performing a collective barrier call to warm up gangs. This can take up to several minutes depending on the topology.")  # fmt: skip

    try:
        gangs.root.barrier()
    except GangError as ex:
        raise RecipeError(
            "The collective barrier after the gang setup operation has failed. See the nested exception for details."  # fmt: skip
        ) from ex

    log.info("Gangs warmed up.")

    return gangs


def _maybe_setup_fsdp_gangs(
    context: RuntimeContext, trainer_section: TrainerSection, gangs: Gangs
) -> Gangs:
    if trainer_section.data_parallelism != "fsdp":
        return gangs

    if trainer_section.fsdp.hybrid:
        if gangs.root.size != gangs.dp.size:  # means we have model parallelism.
            raise HybridShardingNotSupportedError()

        log.info("Initializing hybrid sharded data parallel gangs.")

        try:
            local_world_size = get_local_world_size(context.env)
        except InvalidEnvironmentVariableError as ex:
            raise GangError(
                "The local world size for hybrid sharded data parallelism cannot be determined. See the nested exception for details."
            ) from ex

        if local_world_size == 1:
            log.warning("`trainer.fsdp.hybrid` is set, but the local world size is 1. Skipping the setup of hybrid sharded data parallel gangs.")  # fmt: skip

            return gangs

        if gangs.dp.size % local_world_size != 0:
            raise GangError(
                f"The number of processes in the data parallel gang is expected to be a multiple of the local world size ({local_world_size}) when `trainer.fsdp.hybrid` is set, but is {gangs.dp.size} instead."
            )

        gangs = setup_fsdp_gangs(gangs, local_world_size)

        log.info("Hybrid sharded data parallel gangs initialized.")
    else:
        gangs = setup_fsdp_gangs(gangs)

    return gangs
