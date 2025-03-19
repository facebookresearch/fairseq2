# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from datetime import timedelta

from fairseq2.context import RuntimeContext
from fairseq2.device import DeviceDetectionError, determine_default_device
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
    ConfigSectionNotFoundError,
    GangSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipes.utils.log import log_environment_info
from fairseq2.utils.env import InvalidEnvironmentVariableError, get_local_world_size


def setup_gangs(context: RuntimeContext, recipe_config: object) -> Gangs:
    try:
        device = determine_default_device(context)
    except DeviceDetectionError as ex:
        raise RecipeError(
            "The device of the process cannot be determined. See the nested exception for details."
        ) from ex

    log.info("Setting '{}' as the default device of the process.", device)

    log_environment_info(log, device)

    log.info("Initializing the root gang.")

    gang_section = get_config_section(recipe_config, "gang", GangSection)

    timeout = timedelta(minutes=gang_section.timeout)

    try:
        root_gang = setup_root_gang(
            device,
            timeout=timeout,
            high_priority=gang_section.high_priority,
            monitored=gang_section.monitored,
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

    try:
        gangs = _maybe_setup_fsdp_gangs(context, recipe_config, gangs)
    except GangError as ex:
        raise RecipeError(
            "The hybrid sharded data parallel gangs cannot set up. See the nested exception for details."
        ) from ex

    s = (
        f"Data: {gangs.dp.rank} | "
        f"Data/Replicated: {gangs.rdp.rank} | "
        f"Data/Sharded: {gangs.sdp.rank} | "
        f"Tensor: {gangs.tp.rank}"
    )

    log.info("Process Ranks - {}", s)

    return gangs


def _maybe_setup_fsdp_gangs(
    context: RuntimeContext, recipe_config: object, gangs: Gangs
) -> Gangs:
    try:
        trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)
    except ConfigSectionNotFoundError:
        return gangs

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
