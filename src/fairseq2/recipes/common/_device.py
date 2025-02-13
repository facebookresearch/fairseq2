# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.device import (
    CudaDeviceStatTracker,
    DeviceStatTracker,
    NoopDeviceStatTracker,
)
from fairseq2.gang import Gangs


def create_device_stat_tracker(gangs: Gangs) -> DeviceStatTracker:
    if gangs.root.rank != 0:
        return NoopDeviceStatTracker()

    if gangs.root.device.type == "cuda":
        return CudaDeviceStatTracker(gangs.root.device)

    return NoopDeviceStatTracker()
