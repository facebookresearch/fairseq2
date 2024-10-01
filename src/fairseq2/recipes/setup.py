# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import torch

from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.device import determine_default_device
from fairseq2.gang import Gang, setup_default_gang, setup_parallel_gangs
from fairseq2.logging import get_log_writer
from fairseq2.recipes.config import (
    ConfigurationError,
    ConfigurationManager,
    StandardConfigurationManager,
)
from fairseq2.recipes.utils.log import log_environment_info
from fairseq2.typing import DataClass, Device
from fairseq2.utils.structured import ValueConverter

log = get_log_writer(__name__)


@dataclass
class GangConfig:
    timeout: int = 15
    monitored: bool = False
    tensor_parallel_size: int = 1


def _register_config(container: DependencyContainer, config: DataClass) -> None:
    monitored_gang = getattr(config, "monitored_gang", False)

    tensor_parallel_size = getattr(config, "tensor_parallel_size", 1)

    config_: dict[str, object] = {
        "gang": GangConfig(
            monitored=monitored_gang,
            tensor_parallel_size=tensor_parallel_size,
        ),
    }

    container.register_instance(dict, config_, key="config")


def _register_recipe_objects(container: DependencyContainer) -> None:
    container.register_factory(ConfigurationManager, _create_config_manager)

    container.register_factory(Device, _create_device)

    # Gangs
    container.register_factory(GangConfig, _create_gang_config)

    container.register_factory(Gang, _create_root_gang)

    container.register_factory(Gang, _create_dp_gang, key="dp")
    container.register_factory(Gang, _create_tp_gang, key="tp")


def _create_config_manager(resolver: DependencyResolver) -> ConfigurationManager:
    value_converter = resolver.resolve(ValueConverter)

    unstructured_config = resolver.resolve(dict, key="config")

    return StandardConfigurationManager(value_converter, unstructured_config)


def _create_device(resolver: DependencyResolver) -> Device:
    device = determine_default_device()

    # In case we run on Ampere or later, use TF32.
    torch.set_float32_matmul_precision("high")

    log_environment_info(log, device)

    return device


def _create_gang_config(resolver: DependencyResolver) -> GangConfig:
    config_manager = resolver.resolve(ConfigurationManager)

    config = config_manager.get_optional_section("gang", GangConfig)

    return config or GangConfig()


def _create_root_gang(resolver: DependencyResolver) -> Gang:
    device = resolver.resolve(Device)

    config = resolver.resolve(GangConfig)

    timeout = timedelta(minutes=config.timeout)

    log.info("Initializing the root gang.")

    gang = setup_default_gang(
        device=device, timeout=timeout, monitored=config.monitored
    )

    log.info("Root gang initialized.")

    return gang


def _create_dp_gang(resolver: DependencyResolver) -> Gang:
    root_gang = resolver.resolve(Gang)

    config = resolver.resolve(GangConfig)

    log.info("Initializing data parallel gangs.")

    try:
        gangs = setup_parallel_gangs(
            root_gang, tp_size=config.tensor_parallel_size, setup_only={"dp"}
        )
    except ValueError as ex:
        raise ConfigurationError(
            f"The size of the root gang ({root_gang.size}) is not divisible by `gang.tensor_parallel_size` ({config.tensor_parallel_size})."
        ) from ex

    log.info("Gangs initialized.")

    return gangs["dp"]


def _create_tp_gang(resolver: DependencyResolver) -> Gang:
    root_gang = resolver.resolve(Gang)

    config = resolver.resolve(GangConfig)

    log.info("Initializing tensor parallel gangs.")

    try:
        gangs = setup_parallel_gangs(
            root_gang, tp_size=config.tensor_parallel_size, setup_only={"tp"}
        )
    except ValueError as ex:
        raise ConfigurationError(
            f"The size of the root gang ({root_gang.size}) is not divisible by `gang.tensor_parallel_size` ({config.tensor_parallel_size})."
        ) from ex

    log.info("Gangs initialized.")

    return gangs["tp"]
