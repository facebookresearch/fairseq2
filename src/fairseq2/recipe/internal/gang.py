# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from datetime import timedelta
from typing import final

import torch.distributed as dist

from fairseq2.error import OperationalError
from fairseq2.device import Device
from fairseq2.gang import (
    FakeGang,
    Gang,
    GangError,
    Gangs,
    ProcessGroupGang,
    create_fsdp_gangs,
    create_parallel_gangs,
)
from fairseq2.logging import log
from fairseq2.recipe.config import GangSection, TrainerSection
from fairseq2.recipe.error import ConfigError
from fairseq2.utils.validation import ValidationError
from fairseq2.world_info import WorldInfo


@final
class _GangsFactory:
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
                raise ConfigError(
                    f"Only `cpu` and `cuda` devices are supported, but the device of the process is `{self._device}`."
                )

            if not dist.is_available():
                raise ConfigError("torch.distributed is not available.")

            timeout = timedelta(minutes=section.timeout)

            try:
                root_gang = ProcessGroupGang.create_default_process_group(
                    self._device, timeout=timeout, high_priority=section.high_priority
                )
            except GangError as ex:
                raise OperationalError("Failed to create root gang.") from ex
        else:
            root_gang = FakeGang(self._device)

        log.info("Root gang created.")

        log.info("Creating parallel gangs.")

        tp_size = section.tensor_parallel_size

        if root_gang.size % tp_size != 0:
            raise ConfigError(
                f"`gang.tensor_parallel_size` must be a factor of the number of processes in the root gang ({root_gang.size}), but is {tp_size} instead."
            )

        try:
            gangs = create_parallel_gangs(root_gang, tp_size=tp_size)
        except GangError as ex:
            raise OperationalError("Failed to create parallel gangs.") from ex

        log.info("Parallel gangs created.")

        return gangs


@final
class _FSDPGangsFactory:
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
                raise ConfigError(
                    f"Local world size must be a factor of the number of processes in the data parallel gang ({gangs.dp.size}), but is {local_world_size} instead."
                )

            log.info("Creating hybrid sharded data parallel gangs.")

            try:
                gangs = create_fsdp_gangs(gangs, local_world_size)
            except GangError as ex:
                raise OperationalError("Failed to create FSDP gangs.") from ex

            log.info("Hybrid sharded data parallel gangs created.")
        else:
            try:
                gangs = create_fsdp_gangs(gangs)
            except GangError as ex:
                raise OperationalError("Failed to create FSDP gangs.") from ex

        return gangs


def _warmup_gangs(gangs: Gangs) -> None:
    log.info("Performing a collective barrier call to warm up gangs. This can take up to several minutes depending on the topology.")  # fmt: skip

    try:
        gangs.root.barrier()
    except GangError as ex:
        raise OperationalError("Failed to warm up gangs.") from ex

    log.info("Gangs warmed up.")
