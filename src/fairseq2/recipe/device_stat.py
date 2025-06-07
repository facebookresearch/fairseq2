# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.gang import Gangs
from fairseq2.runtime.provider import Provider
from fairseq2.utils.device_stat import DeviceStatTracker, NoopDeviceStatTracker


@final
class DeviceStatTrackerFactory:
    def __init__(self, trackers: Provider[DeviceStatTracker], gangs: Gangs) -> None:
        self._trackers = trackers
        self._gangs = gangs

    def create(self) -> DeviceStatTracker:
        if self._gangs.root.rank == 0:
            return self._trackers.get(self._gangs.root.device.type)

        return NoopDeviceStatTracker()
