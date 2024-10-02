# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.gang import Gang, setup_default_gang, setup_parallel_gangs
from fairseq2.logging import get_log_writer
from fairseq2.recipes.config_manager import ConfigError, register_config
from fairseq2.typing import Device

log = get_log_writer(__name__)


@dataclass
class GangConfig:
    timeout: int = 15
    monitored: bool = False
    tensor_parallel_size: int = 1


def register_gangs(container: DependencyContainer) -> None:
    register_config(container, path="gang", kls=GangConfig, default_factory=GangConfig)

    container.register_factory(Gang, _create_root_gang)

    container.register_factory(Gang, _create_dp_gang, key="dp")
    container.register_factory(Gang, _create_tp_gang, key="tp")


def _create_root_gang(resolver: DependencyResolver) -> Gang:
    device = resolver.resolve(Device)

    config = resolver.resolve(GangConfig, key="gang")

    timeout = timedelta(minutes=config.timeout)

    log.info("Initializing the root gang.")

    gang = setup_default_gang(
        device=device, timeout=timeout, monitored=config.monitored
    )

    log.info("Root gang initialized.")

    return gang


def _create_dp_gang(resolver: DependencyResolver) -> Gang:
    root_gang = resolver.resolve(Gang)

    config = resolver.resolve(GangConfig, key="gang")

    log.info("Initializing data parallel gangs.")

    try:
        gangs = setup_parallel_gangs(
            root_gang, tp_size=config.tensor_parallel_size, setup_only={"dp"}
        )
    except ValueError as ex:
        raise ConfigError(
            f"The size of the root gang ({root_gang.size}) is not divisible by `gang.tensor_parallel_size` ({config.tensor_parallel_size})."
        ) from ex

    log.info("Gangs initialized.")

    return gangs["dp"]


def _create_tp_gang(resolver: DependencyResolver) -> Gang:
    root_gang = resolver.resolve(Gang)

    config = resolver.resolve(GangConfig, key="gang")

    log.info("Initializing tensor parallel gangs.")

    try:
        gangs = setup_parallel_gangs(
            root_gang, tp_size=config.tensor_parallel_size, setup_only={"tp"}
        )
    except ValueError as ex:
        raise ConfigError(
            f"The size of the root gang ({root_gang.size}) is not divisible by `gang.tensor_parallel_size` ({config.tensor_parallel_size})."
        ) from ex

    log.info("Gangs initialized.")

    return gangs["tp"]
