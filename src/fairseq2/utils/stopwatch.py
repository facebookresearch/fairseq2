# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum
from time import perf_counter
from typing import Any, final

import torch
from typing_extensions import Self

from fairseq2.error import InvalidOperationError
from fairseq2.typing import CPU, Device


class _StopwatchState(Enum):
    NOT_STARTED = 0
    RUNNING = 1
    PAUSED = 1


@final
class Stopwatch:
    """Measures elapsed execution time."""

    _is_running: bool
    _accumulated_duration: float
    _start_time: float
    _device: Device

    def __init__(self, *, device: Device | None = None) -> None:
        """
        :param device: If not ``None``, waits for all operations on ``device``
            to complete before measuring the elapsed time. Note that this can
            have a negative impact on the runtime performance if not used
            carefully.
        """
        self._is_running = False

        self._accumulated_duration = 0.0

        self._start_time = 0.0

        if device is not None:
            if device.type != "cpu" and device.type != "cuda":
                raise ValueError(
                    f"The type of `device` must be `cpu` or `cuda`, but is `{device.type}` instead."
                )

        self._device = device or CPU

    def start(self) -> None:
        if self._is_running:
            raise InvalidOperationError("The stopwatch is already running.")

        self._start_time = perf_counter()

        self._is_running = True

    def stop(self) -> None:
        if not self._is_running:
            return

        self._maybe_sync_device()

        self._accumulated_duration += perf_counter() - self._start_time

        self._is_running = False

    def reset(self) -> None:
        self._accumulated_duration = 0.0

        if self._is_running:
            self._maybe_sync_device()

            self._start_time = perf_counter()

    def get_elapsed_time(self) -> float:
        if not self._is_running:
            return self._accumulated_duration

        self._maybe_sync_device()

        return self._accumulated_duration + (perf_counter() - self._start_time)

    def _maybe_sync_device(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def __enter__(self) -> Self:
        self.start()

        return self

    def __exit__(self, *ex: Any) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._is_running
