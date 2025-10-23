# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides abstractions for managing PyTorch devices and handling CUDA
contexts.
"""

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
    """Represents an object that can be transferred between devices."""

    @abstractmethod
    def to(self, device: Device, *, non_blocking: bool = False) -> None:
        """
        Transfers this object to the specified device.

        If ``non_blocking`` is ``True``, the transfer will be performed
        asynchronously if possible.
        """


def get_default_device() -> Device:
    """
    Returns the default device of this process.

    The default device is determined by the following precedence:

    1) If ``FAIRSEQ2_DEVICE`` environment variable is set, the specified device
       will be used.
    2) If CUDA is enabled and ``CUDA_VISIBLE_DEVICES`` environment variable
       contains a single device, the specified device will be used.
    3) If CUDA is enabled and ``LOCAL_RANK`` environment variable is set, the
       CUDA device at the specified index will be used.
    4) CPU will be used.

    :raises LocalRankOutOfRangeError: If ``LOCAL_RANK`` environment variable is
        less than zero or exceeds the number of available devices.
    """
    resolver = get_dependency_resolver()

    return resolver.resolve(Device)


def get_current_device() -> Device:
    """
    Returns the current device of the calling thread.

    The current device of a thread can be changed by using a device as a context
    manager. See `here`__ for more information.

    .. __: https://docs.pytorch.org/tutorials/recipes/recipes/changing_default_device.html

    .. note::

        PyTorch does not currently expose a public API to retrieve the current
        device of a thread. If such an API becomes available in the future, this
        function will act as an alias for it.

    .. warning::

        This function might impose a slight performance cost. Avoid calling it
        in hot code paths.

    .. code:: python

        import torch

        from fairseq2.device import get_current_device

        # Default device used by PyTorch. Typically CPU.
        default_device = torch.get_default_device()

        assert get_current_device() == default_device

        device = torch.device("cuda:0")

        # Instruct PyTorch to use the specified device instead of the default
        # device for tensor factory operations.
        with device:
            assert get_current_device() == device
    """
    # One might expect `torch.get_default_device()` to return the nearest device
    # set as context in the call stack; however, this is not the case. Currently,
    # the only way to determine the current device is to create a dummy tensor
    # and inspect its device attribute.
    return torch.empty(()).device


@final
class DefaultDeviceDetector:
    """This class is used internally by the :func:`get_default_device` function."""

    def __init__(
        self, env: Environment, world_info: WorldInfo, cuda_context: CudaContext
    ) -> None:
        self._env = env
        self._world_info = world_info
        self._cuda_context = cuda_context

    def detect(self) -> Device:
        """
        Attemps to detect a default device. If ``FAIRSEQ2_DEVICE`` is set, use
        indicated device

        1) If env variable not set, attempt to get default CUDA device
        2) If no CUDA device is found, set device to CPU
        """
        device = self._maybe_get_device_from_env("FAIRSEQ2_DEVICE")

        if device is None:
            device = self._get_default_cuda_device()

        if device is None:
            device = CPU

        return device

    def _maybe_get_device_from_env(self, var_name: str) -> Device | None:
        """Refer to :func:get_default_device."""
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
        """
        Retrieve the system's default CUDA device
        By default, this is device at index 0 (first listed cuda device)

        1) If there is no cuda context, return None (no device available)
        2) If number of devices is 0, return None (no device available)
        3) If at least one device available, get number of visible devices
           :raises a ValueError: if devices stored as a list
        4) By default, grab device at index 0 and set device to CUDA
        5) If multiple devices, get first device with CUDA index
        """
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
        """
        Get index of a device

        1) If no devices, :raises an InternalError:
        2) Else, get number of devices available as local_rank
        3) If local_rank is greater than num declared devices,
           :raises a LocalRankOutOfRangeError: (rank is outside
           of num_devices index bounds)
        """
        if num_devices <= 0:
            raise InternalError(f"`num_devices` is {num_devices}.")

        local_rank = self._world_info.local_rank

        if local_rank >= num_devices:
            raise LocalRankOutOfRangeError(local_rank, num_devices, device_type)

        return local_rank


class LocalRankOutOfRangeError(Exception):
    """
    Raised by :func:`get_default_device` when ``LOCAL_RANK`` environment variable
    is less than zero or exceeds the number of available devices.
    """

    def __init__(self, local_rank: int, num_devices: int, device_type: str) -> None:
        super().__init__(
            f"Host has {num_devices} {device_type} device(s), but the local rank of the process is {local_rank}."
        )

        self.local_rank = local_rank
        self.num_devices = num_devices
        self.device_type = device_type


class CudaContext(ABC):
    """
    Represents an interface for interacting with CUDA runtime and device
    information.
    """

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
    """Represents the standard implementation of :class:`CudaContext`."""

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
