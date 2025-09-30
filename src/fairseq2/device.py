# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final, TypeAlias, final

import torch
from typing_extensions import override

from fairseq2.error import InternalError
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.utils.env import Environment, EnvironmentVariableError
from fairseq2.world_info import WorldInfo

Device: TypeAlias = torch.device


CPU: Final = Device("cpu")

META_DEVICE: Final = Device("meta")


class SupportsDeviceTransfer(ABC):
    @abstractmethod
    def to(self, device: Device, *, non_blocking: bool = False) -> None: ...


def get_default_device() -> Device:
    resolver = get_dependency_resolver()

    return resolver.resolve(Device)


@final
class DefaultDeviceDetector:
    def __init__(
        self, env: Environment, world_info: WorldInfo, cuda_context: CudaContext
    ) -> None:
        self._env = env
        self._world_info = world_info
        self._cuda_context = cuda_context

    def detect(self) -> Device:
        device = self._maybe_get_device_from_env("FAIRSEQ2_DEVICE")

        if device is None:
            device = self._get_default_cuda_device()

        if device is None:
            device = CPU

        return device

    def _maybe_get_device_from_env(self, var_name: str) -> Device | None:
        s = self._env.maybe_get(var_name)
        if s is None:
            return None

        try:
            return Device(s)
        except (RuntimeError, ValueError) as ex:
            msg = (
                f"{var_name} environment variable cannot be parsed as a PyTorch device."
            )

            raise EnvironmentVariableError(var_name, msg) from ex

    def _get_default_cuda_device(self) -> Device | None:
        if not self._cuda_context.is_available():
            return None

        num_devices = self._cuda_context.device_count()
        if num_devices == 0:
            return None

        visible_devices = self._env.maybe_get("CUDA_VISIBLE_DEVICES")
        if visible_devices is not None:
            try:
                int(visible_devices)
            except ValueError:
                # If here, this means CUDA_VISIBLE_DEVICES is a list instead of
                # a single device index.
                device = None
            else:
                device = Device("cuda", index=0)
        else:
            device = None

        if device is None:
            idx = self._get_device_index(num_devices, device_type="cuda")

            device = Device("cuda", index=idx)

        return device

    def _get_device_index(self, num_devices: int, device_type: str) -> int:
        if num_devices <= 0:
            raise InternalError(f"`num_devices` is {num_devices}.")

        local_rank = self._world_info.local_rank

        if local_rank >= num_devices:
            raise LocalRankOutOfRangeError(local_rank, num_devices, device_type)

        return local_rank


class LocalRankOutOfRangeError(Exception):
    def __init__(self, local_rank: int, num_devices: int, device_type: str) -> None:
        super().__init__(
            f"Host has {num_devices} {device_type} device(s), but the local rank of the process is {local_rank}."
        )

        self.local_rank = local_rank
        self.num_devices = num_devices
        self.device_type = device_type


class CudaContext(ABC):
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def device_count(self) -> int: ...

    @abstractmethod
    def get_device_properties(self, device: Device) -> Any: ...

    @abstractmethod
    def memory_stats(self, device: Device) -> dict[str, Any]: ...

    @abstractmethod
    def reset_peak_memory_stats(self) -> None: ...


@final
class StandardCudaContext(CudaContext):
    @override
    def is_available(self) -> bool:
        return torch.cuda.is_available()

    @override
    def device_count(self) -> int:
        return torch.cuda.device_count()

    @override
    def get_device_properties(self, device: Device) -> Any:
        return torch.cuda.get_device_properties(device)

    @override
    def memory_stats(self, device: Device) -> dict[str, Any]:
        return torch.cuda.memory_stats(device)

    @override
    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()
