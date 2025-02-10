# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from datetime import timedelta

from fairseq2.context import RuntimeContext
from fairseq2.device import DeviceDetectionError, determine_default_device
from fairseq2.error import ProgramError
from fairseq2.gang import GangError, Gangs, setup_parallel_gangs, setup_root_gang
from fairseq2.logging import log
from fairseq2.recipes.config import GangSection, get_config_section
from fairseq2.recipes.utils.log import log_environment_info


def setup_gangs(context: RuntimeContext, recipe_config: object) -> Gangs:
    try:
        device = determine_default_device()
    except DeviceDetectionError as ex:
        raise ProgramError(
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
        raise ProgramError(
            "The root gang of the process cannot be set up. See the nested exception for details."
        ) from ex

    log.info("Root gang initialized.")

    log.info("Initializing the parallel gangs.")

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
        raise ProgramError(
            "The parallel gangs of the process cannot be set up. See the nested exception for details."
        ) from ex

    log.info("Parallel gangs initialized.")

    return gangs
