# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import final

import torch
from typing_extensions import override

from fairseq2.error import InternalError
from fairseq2.typing import CPU, Device
from fairseq2.utils.env import (
    InvalidEnvironmentVariableError,
    get_device_from_env,
    get_int_from_env,
)


@final
class DefaultDeviceAccessor:
    _env: Mapping[str, str]
    _cuda_context: CudaContext

    def __init__(self, env: Mapping[str, str], cuda_context: CudaContext) -> None:
        self._env = env
        self._cuda_context = cuda_context

    def get(self) -> Device:
        try:
            device = get_device_from_env(self._env, "FAIRSEQ2_DEVICE")
        except InvalidEnvironmentVariableError as ex:
            raise DeviceDetectionError(
                "The default device cannot be set using the `FAIRSEQ2_DEVICE` environment variable. See the nested exception for details."
            ) from ex

        if device is None:
            device = self._determine_default_cuda_device()

        if device is None:
            device = CPU

        if device.type == "cuda":
            self._cuda_context.set_default_device(device)

        return device

    def _determine_default_cuda_device(self) -> Device | None:
        if not self._cuda_context.is_available():
            return None

        num_devices = self._cuda_context.device_count()
        if num_devices == 0:
            return None

        visible_devices = self._env.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is not None:
            try:
                int(visible_devices)
            except ValueError:
                # If we are here, it means CUDA_VISIBLE_DEVICES is a list instead of
                # a single device index.
                device = None
            else:
                device = Device("cuda", index=0)
        else:
            device = None

        if device is None:
            try:
                idx = self._get_device_index(num_devices, device_type="cuda")
            except InvalidEnvironmentVariableError as ex:
                raise DeviceDetectionError(
                    "The default `cuda` device index cannot be inferred from the environment. See the nested exception for details."
                ) from ex

            device = Device("cuda", index=idx)

        return device

    def _get_device_index(self, num_devices: int, device_type: str) -> int:
        if num_devices <= 0:
            raise InternalError(f"`num_devices` is {num_devices}.")

        # We use the `LOCAL_RANK` environment variable to determine which device to
        # pick in case the process has more than one available.
        device_idx = get_int_from_env(self._env, "LOCAL_RANK", allow_zero=True)
        if device_idx is None:
            num_procs = get_int_from_env(self._env, "LOCAL_WORLD_SIZE")
            if num_procs is not None and num_procs > 1 and num_devices > 1:
                raise InvalidEnvironmentVariableError(
                    "LOCAL_RANK", f"The default `{device_type}` device cannot be determined. There are {num_devices} devices available, but the `LOCAL_RANK` environment variable is not set."  # fmt: skip
                )

            return 0

        if device_idx < 0:
            raise InvalidEnvironmentVariableError(
                "LOCAL_RANK", f"The value of the `LOCAL_RANK` environment variable is expected to be greater than or equal to 0, but is {device_idx} instead."  # fmt: skip
            )

        if device_idx >= num_devices:
            raise InvalidEnvironmentVariableError(
                "LOCAL_RANK", f"The value of the `LOCAL_RANK` environment variable is expected to be less than the number of available `{device_type}` devices ({num_devices}), but is {device_idx} instead."  # fmt: skip
            )

        return device_idx


class DeviceDetectionError(Exception):
    pass


def determine_default_device() -> Device:
    cuda_context = TorchCudaContext()

    device_accessor = DefaultDeviceAccessor(os.environ, cuda_context)

    return device_accessor.get()


class CudaContext(ABC):
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def device_count(self) -> int: ...

    @abstractmethod
    def set_default_device(self, device: Device) -> None: ...


@final
class TorchCudaContext(CudaContext):
    @override
    def is_available(self) -> bool:
        return torch.cuda.is_available()

    @override
    def device_count(self) -> int:
        return torch.cuda.device_count()

    @override
    def set_default_device(self, device: Device) -> None:
        torch.cuda.set_device(device)
