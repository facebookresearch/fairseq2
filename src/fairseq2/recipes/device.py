# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.device import determine_default_device
from fairseq2.logging import get_log_writer
from fairseq2.recipes.utils.log import log_environment_info
from fairseq2.typing import Device

log = get_log_writer(__name__)


def register_device(container: DependencyContainer) -> None:
    container.register_factory(Device, _create_default_device)


def _create_default_device(resolver: DependencyResolver) -> Device:
    device = determine_default_device()

    # In case we run on Ampere or later, use TF32.
    torch.set_float32_matmul_precision("high")

    log_environment_info(log, device)

    return device
