# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from datetime import timedelta

import torch.distributed as dist

from fairseq2.device import Device
from fairseq2.gang import (
    FakeGang,
    Gang,
    GangError,
    Gangs,
    ProcessGroupGang,
    create_fsdp_gangs,
    create_parallel_gangs,
    raise_operational_gang_error,
)
from fairseq2.logging import log
from fairseq2.recipe.config import (
    ConfigSectionNotFoundError,
    GangSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.log import log_ranks
from fairseq2.utils.validation import ValidationError
from fairseq2.world_info import WorldInfo


def _create_gangs(resolver: DependencyResolver) -> Gangs:
    gangs = _create_parallel_gangs(resolver)

    gangs = _maybe_create_fsdp_gangs(resolver, gangs)

    log.info("Performing a collective barrier call to warm up gangs. This can take up to several minutes depending on the topology.")  # fmt: skip

    try:
        gangs.root.barrier()
    except GangError as ex:
        raise_operational_gang_error(ex)

    log.info("Gangs warmed up.")

    log_ranks(gangs)

    return gangs


def _create_parallel_gangs(resolver: DependencyResolver) -> Gangs:
    gang_section = get_config_section(resolver, "gang", GangSection)

    device = resolver.resolve(Device)

    world_info = resolver.resolve(WorldInfo)

    log.info("Creating the root gang.")

    root_gang: Gang

    if world_info.size > 1:
        if device.type != "cpu" and device.type != "cuda":
            raise DeviceTypeNotSupportedError(device)

        if not dist.is_available():
            raise TorchDistributedNotAvailableError()

        timeout = timedelta(minutes=gang_section.timeout)

        try:
            root_gang = ProcessGroupGang.create_default_process_group(
                device, timeout=timeout, high_priority=gang_section.high_priority
            )
        except GangError as ex:
            raise_operational_gang_error(ex)
    else:
        root_gang = FakeGang(device)

    log.info("Root gang created.")

    log.info("Creating parallel gangs.")

    tp_size = gang_section.tensor_parallel_size

    if root_gang.size % tp_size != 0:
        raise GangTopologyError(root_gang.size, tp_size)

    try:
        gangs = create_parallel_gangs(root_gang, tp_size=tp_size)
    except GangError as ex:
        raise_operational_gang_error(ex)

    log.info("Parallel gangs created.")

    return gangs


class DeviceTypeNotSupportedError(Exception):
    def __init__(self, device: Device) -> None:
        super().__init__(
            f"For distributed jobs, only `cpu` and `cuda` devices are supported, but the device of the process is `{device}`."
        )

        self.device = device
        self.supported_devices = {"cpu", "cuda"}


class TorchDistributedNotAvailableError(Exception):
    def __init__(self) -> None:
        super().__init__("torch.distributed is not available.")


class GangTopologyError(Exception):
    def __init__(self, world_size: int, tp_size: int) -> None:
        super().__init__(
            f"`gang.tensor_parallel_size` must be a factor of the number of processes in the gang ({world_size}), but is {tp_size} instead."
        )

        self.world_size = world_size
        self.tp_size = tp_size


def _maybe_create_fsdp_gangs(resolver: DependencyResolver, gangs: Gangs) -> Gangs:
    try:
        trainer_section = get_config_section(resolver, "trainer", TrainerSection)
    except ConfigSectionNotFoundError:
        trainer_section = None

    if trainer_section is None:
        return gangs

    if trainer_section.data_parallelism != "fsdp":
        return gangs

    world_info = resolver.resolve(WorldInfo)

    if trainer_section.fsdp.hybrid:
        if gangs.root.size != gangs.dp.size:  # means we have model parallelism.
            raise ValidationError(
                "`trainer.fsdp.hybrid` is set, but HSDP is not supported when `gang.tensor_parallel_size` is greater than 1."
            )

        local_world_size = world_info.local_size

        if local_world_size == 1:
            log.warning("`trainer.fsdp.hybrid` is set, but the local world size is 1. Hybrid sharded data parallelism won't be in effect.")  # fmt: skip

            return gangs

        if gangs.dp.size % local_world_size != 0:
            raise HSDPTopologyError(local_world_size, gangs.dp.size)

        log.info("Creating hybrid sharded data parallel gangs.")

        try:
            gangs = create_fsdp_gangs(gangs, local_world_size)
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Hybrid sharded data parallel gangs created.")
    else:
        try:
            gangs = create_fsdp_gangs(gangs)
        except GangError as ex:
            raise_operational_gang_error(ex)

    return gangs


class HSDPTopologyError(Exception):
    def __init__(self, local_world_size: int, dp_size: int) -> None:
        super().__init__(
            f"Local world size must be a factor of the number of processes in the data parallel gang ({dp_size}), but is {local_world_size} instead."
        )

        self.local_world_size = local_world_size
        self.dp_size = dp_size
