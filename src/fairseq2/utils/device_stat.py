# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final, final

from typing_extensions import override

from fairseq2.device import CudaContext, Device
from fairseq2.error import OperationalError


class DeviceStatTracker(ABC):
    @abstractmethod
    def get_stats(self) -> dict[str, object]: ...

    @abstractmethod
    def reset(self) -> None: ...


@final
class _NoopDeviceStatTracker(DeviceStatTracker):
    @override
    def get_stats(self) -> dict[str, object]:
        return {}

    @override
    def reset(self) -> None:
        pass


NOOP_DEVICE_STAT_TRACKER: Final = _NoopDeviceStatTracker()


@final
class CudaDeviceStatTracker(DeviceStatTracker):
    def __init__(self, device: Device, cuda_context: CudaContext) -> None:
        if device.type != "cuda":
            raise ValueError(
                f"`device.type` must be `cuda`, but is `{device.type}` instead."
            )

        self._device = device
        self._cuda_context = cuda_context

        try:
            props = cuda_context.get_device_properties(device)
        except RuntimeError as ex:
            raise OperationalError("CUDA call failed.") from ex

        self._total_memory = props.total_memory

    @override
    def get_stats(self) -> dict[str, object]:
        try:
            stats = self._cuda_context.memory_stats(self._device)
        except RuntimeError as ex:
            raise OperationalError("CUDA call failed.") from ex

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
        try:
            self._cuda_context.reset_peak_memory_stats()
        except RuntimeError as ex:
            raise OperationalError("CUDA call failed.") from ex
