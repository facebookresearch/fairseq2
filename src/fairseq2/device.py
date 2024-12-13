# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

import torch

from fairseq2.error import InternalError
from fairseq2.logging import log
from fairseq2.typing import CPU, Device
from fairseq2.utils.env import (
    InvalidEnvironmentVariableError,
    get_device_from_env,
    get_int_from_env,
)

_default_device: Device | None = None


def determine_default_device() -> Device:
    """Determine the default ``torch.device`` of the process."""
    global _default_device

    if _default_device is not None:
        return _default_device

    try:
        _default_device = get_device_from_env("FAIRSEQ2_DEVICE")
    except InvalidEnvironmentVariableError as ex:
        raise DeviceDetectionError(
            "The default device cannot be set using the `FAIRSEQ2_DEVICE` environment variable. See the nested exception for details."
        ) from ex

    if _default_device is None:
        _default_device = determine_default_cuda_device()

    if _default_device is None:
        _default_device = CPU

    if _default_device.type == "cuda":
        torch.cuda.set_device(_default_device)

    log.info("Setting '{}' as the default device of the process.", _default_device)

    return _default_device


def determine_default_cuda_device() -> Device | None:
    """Determine the default CUDA ``torch.device`` of the process."""
    if not torch.cuda.is_available():
        return None

    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        return None

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
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
            idx = _get_device_index(num_devices, device_type="cuda")
        except InvalidEnvironmentVariableError as ex:
            raise DeviceDetectionError(
                "The default `cuda` device index cannot be inferred from the environment. See the nested exception for details."
            ) from ex

        device = Device("cuda", index=idx)

    return device


def _get_device_index(num_devices: int, device_type: str) -> int:
    if num_devices <= 0:
        raise InternalError(f"`num_devices` is {num_devices}.")

    # We use the `LOCAL_RANK` environment variable to determine which device to
    # pick in case the process has more than one available.
    device_idx = get_int_from_env("LOCAL_RANK", allow_zero=True)
    if device_idx is None:
        num_procs = get_int_from_env("LOCAL_WORLD_SIZE")
        if num_procs is not None and num_procs > 1 and num_devices > 1:
            raise InvalidEnvironmentVariableError(
                f"The default `{device_type}` device cannot be determined. There are {num_devices} devices available, but the `LOCAL_RANK` environment variable is not set."
            )

        return 0

    if device_idx < 0:
        raise InvalidEnvironmentVariableError(
            f"The value of the `LOCAL_RANK` environment variable is expected to be greater than or equal to 0, but is {device_idx} instead."
        )

    if device_idx >= num_devices:
        raise InvalidEnvironmentVariableError(
            f"The value of the `LOCAL_RANK` environment variable is expected to be less than the number of available `{device_type}` devices ({num_devices}), but is {device_idx} instead."
        )

    return device_idx


class DeviceDetectionError(Exception):
    pass
