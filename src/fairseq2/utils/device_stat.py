# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

import torch
from typing_extensions import override

from fairseq2.device import Device


class DeviceStatTracker(ABC):
    @abstractmethod
    def get_stats(self) -> dict[str, object]: ...

    @abstractmethod
    def reset(self) -> None: ...


@final
class NoopDeviceStatTracker(DeviceStatTracker):
    @override
    def get_stats(self) -> dict[str, object]:
        return {}

    @override
    def reset(self) -> None:
        pass


@final
class CudaDeviceStatTracker(DeviceStatTracker):
    _device: Device
    _total_memory: int

    def __init__(self, device: Device) -> None:
        self._device = device

        props = torch.cuda.get_device_properties(device)

        self._total_memory = props.total_memory

    @override
    def get_stats(self) -> dict[str, object]:
        stats = torch.cuda.memory_stats(self._device)

        peak_active_mem_bytes = stats["active_bytes.all.peak"]
        peak_active_mem_ratio = peak_active_mem_bytes / self._total_memory

        peak_reserved_mem_bytes = stats["reserved_bytes.all.peak"]
        peak_reserved_mem_ratio = peak_reserved_mem_bytes / self._total_memory

        return {
            "peak_active_mem_bytes": peak_active_mem_bytes,
            "peak_active_mem_ratio": peak_active_mem_ratio,
            "peak_reserved_mem_bytes": peak_reserved_mem_bytes,
            "peak_reserved_mem_ratio": peak_reserved_mem_ratio,
        }

    @override
    def reset(self) -> None:
        torch.cuda.reset_peak_memory_stats()
