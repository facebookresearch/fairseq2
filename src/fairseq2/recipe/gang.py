# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from datetime import timedelta

from fairseq2.device import Device
from fairseq2.gang import (
    FakeGang,
    Gang,
    Gangs,
    ProcessGroupGang,
    create_fsdp_gangs,
    create_parallel_gangs,
)
from fairseq2.logging import log
from fairseq2.recipe.cluster import WorldInfo
from fairseq2.recipe.config import (
    GangSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipe.error import HybridShardingNotSupportedError
from fairseq2.recipe.utils.log import log_ranks
from fairseq2.runtime.dependency import DependencyResolver


def _create_gangs(resolver: DependencyResolver) -> Gangs:
    gang_section = get_config_section(resolver, "gang", GangSection)

    device = resolver.resolve(Device)

    world_info = resolver.resolve(WorldInfo)

    log.info("Creating the root gang.")

    root_gang: Gang

    if world_info.size > 1:
        timeout = timedelta(minutes=gang_section.timeout)

        root_gang = ProcessGroupGang.create_default_process_group(
            device, timeout=timeout, high_priority=gang_section.high_priority
        )
    else:
        root_gang = FakeGang(device)

    log.info("Root gang created.")

    log.info("Creating parallel gangs.")

    gangs = create_parallel_gangs(root_gang, tp_size=gang_section.tensor_parallel_size)

    log.info("Parallel gangs created.")

    gangs = _maybe_create_fsdp_gangs(resolver, gangs)

    log.info("Performing a collective barrier call to warm up gangs. This can take up to several minutes depending on the topology.")  # fmt: skip

    gangs.root.barrier()

    log.info("Gangs warmed up.")

    log_ranks(gangs)

    return gangs


def _maybe_create_fsdp_gangs(resolver: DependencyResolver, gangs: Gangs) -> Gangs:
    try:
        trainer_section = get_config_section(resolver, "trainer", TrainerSection)
    except LookupError:
        trainer_section = None

    if trainer_section is None:
        return gangs

    if trainer_section.data_parallelism != "fsdp":
        return gangs

    world_info = resolver.resolve(WorldInfo)

    if trainer_section.fsdp.hybrid:
        if gangs.root.size != gangs.dp.size:  # means we have model parallelism.
            raise HybridShardingNotSupportedError()

        if world_info.local_size == 1:
            log.warning("`trainer.fsdp.hybrid` is set, but the local world size is 1. Hybrid sharded data parallelism won't be in effect.")  # fmt: skip

            return gangs

        log.info("Creating hybrid sharded data parallel gangs.")

        gangs = create_fsdp_gangs(gangs, world_info.local_size)

        log.info("Hybrid sharded data parallel gangs created.")
    else:
        gangs = create_fsdp_gangs(gangs)

    return gangs
