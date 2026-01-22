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
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final, TypeAlias, final

import torch
from typing_extensions import override

from fairseq2.error import InternalError
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.typing import ContextManager
from fairseq2.utils.env import Environment, EnvironmentVariableError, get_local_rank
from fairseq2.utils.version import torch_greater_or_equal
from fairseq2.utils.warn import _warn_deprecated

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


def set_device(device: Device) -> ContextManager[None]:
    """
    Changes the device of the calling thread to the specified device.

    This function acts as a context manager, ensuring that within its scope, any
    operation that constructs tensors uses the specified device - unless an
    explicit ``device`` argument is provided.

    .. note::

        This function is equivalent to using a ``torch.device`` as a context
        manager.

    .. code:: python

        import torch

        from fairseq2.device import set_device

        cuda0_device = torch.device("cuda:0")

        with set_device(cuda0_device):
            t = torch.ones((4,4))

            assert t.device == cuda0_device

            cuda1_device = torch.device("cuda:1")

            with set_device(cuda1_device):
                t = torch.ones((4, 4))

                assert t.device == cuda1_device

        t = torch.ones((4, 4))

        assert t.device == torch.device("cpu")
    """
    resolver = get_dependency_resolver()

    return resolver.resolve(DeviceContext).set_device(device)


def get_current_device() -> Device:
    """
    Returns the current device of the calling thread.

    .. warning::

        This function might impose a slight performance cost. Avoid calling it
        in hot code paths.
    """
    resolver = get_dependency_resolver()

    return resolver.resolve(DeviceContext).get_current_device()


class DeviceContext(ABC):
    """
    Provides methods to get and set the current device of the calling thread.

    This interface can be used as an alternative to the corresponding standalone
    functions in object-oriented code.
    """

    @abstractmethod
    def get_current_device(self) -> Device:
        """See :func:`get_current_device`."""

    @abstractmethod
    def set_device(self, device: Device) -> ContextManager[None]:
        """See :func:`set_device`."""


@final
class _StandardDeviceContext(DeviceContext):
    @override
    def get_current_device(self) -> Device:
        if torch_greater_or_equal(2, 8):
            return torch.get_default_device()

        # In PyTorch versions earlier than 2.8 the only way to determine the
        # closest contextual device is to create a dummy tensor.
        return torch.empty(()).device

    @override
    @contextmanager
    def set_device(self, device: Device) -> Iterator[None]:
        with device:
            yield


def detect_default_device() -> Device:
    """
    Detects the default device of this process from environment variables.

    The default device is determined by the following precedence:

    1) If ``FAIRSEQ2_DEVICE`` environment variable is set, the specified device
       will be used.
    2) If CUDA is enabled and ``CUDA_VISIBLE_DEVICES`` environment variable
       contains a single device, the specified device will be used.
    3) If CUDA is enabled and ``LOCAL_RANK`` environment variable is set, the
       CUDA device at the specified index will be used.
    4) CPU will be used.

    :raises EnvironmentVariableError: ``FAIRSEQ2_DEVICE`` environment variable
        does not represent a device.

    :raises EnvironmentVariableError: ``LOCAL_RANK`` environment variable is not
        a positive integer.

    :raises LocalRankOutOfRangeError: ``LOCAL_RANK`` environment variable
        exceeds the number of available devices.
    """
    resolver = get_dependency_resolver()

    return resolver.resolve(_DefaultDeviceDetector).detect()


def get_default_device() -> Device:
    _warn_deprecated(
        "`get_default_device()` is deprecated and will be removed in v0.14. Use `detect_default_device()` instead."
    )

    return detect_default_device()


@final
class _DefaultDeviceDetector:
    def __init__(self, env: Environment, cuda_context: CudaContext) -> None:
        self._env = env
        self._cuda_context = cuda_context

    def detect(self) -> Device:
        """See :func:`detect_default_device`."""
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
            message = f"`{var_name}` environment variable does not represent a PyTorch device."

            raise EnvironmentVariableError(var_name, message) from ex

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

        rank = get_local_rank(self._env)
        if rank is None:
            return 0

        if rank >= num_devices:
            raise LocalRankOutOfRangeError(rank, num_devices, device_type)

        return rank


class LocalRankOutOfRangeError(Exception):
    def __init__(self, rank: int, num_devices: int, device_type: str) -> None:
        super().__init__(
            f"Host has {num_devices} `{device_type}` device(s), but the local rank of the process is {rank}."
        )

        self.rank = rank
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
class _StandardCudaContext(CudaContext):
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
