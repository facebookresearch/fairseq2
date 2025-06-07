# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.device import Device, determine_default_device
from fairseq2.gang import Gangs
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.device_stat import (
    CudaDeviceStatTracker,
    DeviceStatTracker,
    NoopDeviceStatTracker,
)
from fairseq2.utils.env import get_env


def _create_device(resolver: DependencyResolver) -> Device:
    env = get_env(resolver)

    return determine_default_device(env)


def _create_device_stat_tracker(resolver: DependencyResolver) -> DeviceStatTracker:
    gangs = resolver.resolve(Gangs)

    if gangs.root.rank == 0:
        if gangs.root.device.type == "cuda":
            return CudaDeviceStatTracker(gangs.root.device)

    return NoopDeviceStatTracker()
