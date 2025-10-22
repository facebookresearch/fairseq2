# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import final

from typing_extensions import override

from fairseq2.device import Device
from fairseq2.error import OperationalError
from fairseq2.utils.device_stat import DeviceStatTracker

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


@final
class CpuDeviceStatTracker(DeviceStatTracker):
    """Tracks CPU and memory statistics for the current process."""

    def __init__(self, device: Device) -> None:
        if device.type != "cpu":
            raise ValueError(
                f"`device.type` must be `cpu`, but is `{device.type}` instead."
            )

        if psutil is None:
            raise OperationalError(
                "psutil is not installed. Install it with: pip install psutil"
            )

        self._device = device
        self._process = psutil.Process(os.getpid())

        self._peak_memory_rss = 0
        self._peak_memory_vms = 0

    @override
    def get_stats(self) -> dict[str, object]:
        try:
            mem_info = self._process.memory_info()
            mem_percent = self._process.memory_percent()

            current_rss = mem_info.rss
            current_vms = mem_info.vms

            self._peak_memory_rss = max(self._peak_memory_rss, current_rss)
            self._peak_memory_vms = max(self._peak_memory_vms, current_vms)

            stats = {
                "peak_memory_rss_bytes": self._peak_memory_rss,
                "peak_memory_vms_bytes": self._peak_memory_vms,
                "memory_percent": mem_percent,
                "num_threads": self._process.num_threads(),
            }

            cpu_percent = self._process.cpu_percent(interval=None)
            if cpu_percent is not None and cpu_percent > 0:
                stats["cpu_percent"] = cpu_percent

            try:
                load_avg = os.getloadavg()
                stats["load_average_1m"] = load_avg[0]
                stats["load_average_5m"] = load_avg[1]
                stats["load_average_15m"] = load_avg[2]
            except (AttributeError, OSError):
                pass

            return stats

        except Exception as ex:
            raise OperationalError("Failed to collect CPU statistics.") from ex

    @override
    def reset(self) -> None:
        self._peak_memory_rss = 0
        self._peak_memory_vms = 0
        try:
            self._process.cpu_percent(interval=None)
        except Exception:
            pass
