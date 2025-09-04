# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from datetime import timedelta
from typing import final

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
from fairseq2.recipe.config import GangSection, TrainerSection
from fairseq2.recipe.error import (
    DeviceTypeNotSupportedError,
    GangTopologyError,
    HSDPTopologyError,
    TorchDistributedNotAvailableError,
)
from fairseq2.utils.validation import ValidationError
from fairseq2.world_info import WorldInfo


@final
class _RecipeGangsFactory:
    def __init__(
        self, section: GangSection, world_info: WorldInfo, device: Device
    ) -> None:
        self._section = section
        self._world_info = world_info
        self._device = device

    def create(self) -> Gangs:
        log.info("Creating the root gang.")

        section = self._section

        root_gang: Gang

        if self._world_info.size > 1:
            if self._device.type != "cpu" and self._device.type != "cuda":
                raise DeviceTypeNotSupportedError(self._device)

            if not dist.is_available():
                raise TorchDistributedNotAvailableError()

            timeout = timedelta(minutes=section.timeout)

            try:
                root_gang = ProcessGroupGang.create_default_process_group(
                    self._device, timeout=timeout, high_priority=section.high_priority
                )
            except GangError as ex:
                raise_operational_gang_error(ex)
        else:
            root_gang = FakeGang(self._device)

        log.info("Root gang created.")

        log.info("Creating parallel gangs.")

        tp_size = section.tensor_parallel_size

        if root_gang.size % tp_size != 0:
            raise GangTopologyError(root_gang.size, tp_size)

        try:
            gangs = create_parallel_gangs(root_gang, tp_size=tp_size)
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Parallel gangs created.")

        return gangs


@final
class _RecipeFSDPGangsFactory:
    def __init__(self, section: TrainerSection, world_info: WorldInfo) -> None:
        self._section = section
        self._world_info = world_info

    def create(self, gangs: Gangs) -> Gangs:
        if self._section.data_parallelism != "fsdp":
            return gangs

        if self._section.fsdp.hybrid:
            if gangs.root.size != gangs.dp.size:  # means we have model parallelism.
                raise ValidationError(
                    "`trainer.fsdp.hybrid` is set, but HSDP is not supported when `gang.tensor_parallel_size` is greater than 1."
                )

            local_world_size = self._world_info.local_size

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


def _warmup_gangs(gangs: Gangs) -> None:
    log.info("Performing a collective barrier call to warm up gangs. This can take up to several minutes depending on the topology.")  # fmt: skip

    try:
        gangs.root.barrier()
    except GangError as ex:
        raise_operational_gang_error(ex)

    log.info("Gangs warmed up.")


def _log_ranks(gangs: Gangs) -> None:
    s = (
        f"World: {gangs.root.rank}/{gangs.root.size} | "
        f"Data: {gangs.dp.rank}/{gangs.dp.size} | "
        f"Data (Replicated): {gangs.rdp.rank}/{gangs.rdp.size} | "
        f"Data (Sharded): {gangs.sdp.rank}/{gangs.sdp.size} | "
        f"Tensor: {gangs.tp.rank}/{gangs.tp.size} | "
        f"Pipeline: {gangs.pp.rank}/{gangs.pp.size}"
    )

    log.info("Process Ranks - {}", s)
