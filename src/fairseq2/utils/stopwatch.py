# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from time import perf_counter
from typing import Any, final

import torch
from typing_extensions import Self

from fairseq2.error import InvalidOperationError
from fairseq2.typing import Device


@final
class Stopwatch:
    """Measures elapsed execution time."""

    _start_time: float | None
    _device: Device | None

    def __init__(self, *, start: bool = False, device: Device | None = None) -> None:
        """
        :param start: If ``True``, starts the stopwatch immediately.
        :param device: If not ``None``, waits for all operations on ``device``
            to complete before measuring the elapsed time. Note that this can
            have a negative impact on the runtime performance if not used
            carefully.
        """
        self._start_time = None

        if device is not None:
            if device.type != "cpu" and device.type != "cuda":
                raise ValueError(
                    f"The type of `device` must be `cpu` or `cuda`, but is `{device.type}` instead."
                )

        self._device = device

        if start:
            self.start()

    def start(self) -> None:
        """Start the stopwatch."""
        if self._start_time is not None:
            raise InvalidOperationError("The stopwatch is already running.")

        self._sync_device()

        self._start_time = perf_counter()

    def stop(self) -> None:
        """Stop the stopwatch."""
        self._start_time = None

    def reset(self) -> None:
        """Reset the stopwatch."""
        if self._start_time is None:
            raise InvalidOperationError("The stopwatch is not running.")

        self._sync_device()

        self._start_time = perf_counter()

    def get_elapsed_time(self) -> float:
        """Return the elapsed time since the last :meth:`start` or :meth:`reset`."""
        if self._start_time is None:
            return 0.0

        self._sync_device()

        return perf_counter() - self._start_time

    def _sync_device(self) -> None:
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def __enter__(self) -> Self:
        if self._start_time is None:
            self.start()

        return self

    def __exit__(self, *ex: Any) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the stopwatch is running."""
        return self._start_time is not None
