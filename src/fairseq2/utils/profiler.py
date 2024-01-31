# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
from logging import Logger
from time import perf_counter
from typing import Any, Optional

import psutil
import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from typing_extensions import Self

from fairseq2.data.typing import PathLike
from fairseq2.gang import Gang
from fairseq2.typing import Device


class Profiler:
    """Represents a convenience wrapper for :class:`profile`."""

    _profile: Optional[profile]

    def __init__(
        self,
        skip_first: int,
        active: int,
        log_dir: PathLike,
        gang: Gang,
        enabled: bool = False,
    ) -> None:
        """
        :param skip_first:
            The number of steps to skip at the beginning. The last skipped step
            will be treated as the warm-up step.
        :param active:
            The number of steps with active recording.
        :param log_dir:
            The TensorBoard log directory under which to store the trace files.
        :param gang:
            The associated gang.
        :param enabled:
            If ``False``, skips recording and becomes a no-op.
        """
        if skip_first <= 0:
            raise ValueError("`skip_first` must be greater than zero.")

        if not enabled:
            self._profile = None

            return

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        schedule_ = schedule(
            skip_first=skip_first - 1, wait=0, warmup=1, active=active, repeat=1
        )

        trace_handler = tensorboard_trace_handler(
            str(log_dir), worker_name=f"rank_{gang.rank}"
        )

        self._profile = profile(
            activities=activities,
            schedule=schedule_,
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
        )

    def start(self) -> None:
        """Start profiling."""
        if self._profile is not None:
            self._profile.start()

    def stop(self) -> None:
        """Stop profiling."""
        if self._profile is not None:
            self._profile.stop()

    def step(self) -> None:
        """Move to the next profiling step."""
        if self._profile is not None:
            self._profile.step()

    def __enter__(self) -> Self:
        self.start()

        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def wrapped_profile(self) -> Optional[profile]:
        """The wrapped :class:`profile` instance."""
        return self._profile


class Stopwatch:
    """Measures elapsed execution time."""

    start_time: Optional[float]
    device: Optional[Device]

    def __init__(self, *, start: bool = False, device: Optional[Device] = None) -> None:
        """
        :param start:
            If ``True``, starts the stopwatch immediately.
        :param device:
            If specified, waits for all operations on ``device`` to complete
            before measuring the elapsed time. Note that this can have a
            negative impact on the runtime performance if not used carefully.
        """
        self.start_time = None
        self.device = device

        if start:
            self.start()

    def start(self) -> None:
        """Start the stopwatch."""
        if self.start_time is not None:
            raise RuntimeError("The stopwatch is already running.")

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self.start_time = perf_counter()

    def stop(self) -> None:
        """Stop the stopwatch."""
        self.start_time = None

    def get_elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        return perf_counter() - self.start_time

    def __enter__(self) -> Self:
        if self.start_time is None:
            self.start()

        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the stopwatch is running."""
        return self.start_time is not None


def log_environment_info(logger: Logger, device: Optional[Device] = None) -> None:
    """Log information about the software and hardware environments."""
    log_software_info(logger, device)
    log_hardware_info(logger, device)


def log_software_info(logger: Logger, device: Optional[Device] = None) -> None:
    """Log information about the software environment."""
    if not logger.isEnabledFor(logging.INFO):
        return

    info = []

    info.append(f"PyTorch: {torch.__version__}")

    if device is not None and device.type == "cuda":
        info.append(f"CUDA: {torch.version.cuda}")

    info.append(f"Intraop Thread Count: {torch.get_num_threads()}")

    s = " | ".join(info)

    logger.info(f"Software Info - {s}")


def log_hardware_info(logger: Logger, device: Optional[Device] = None) -> None:
    """Log information about the host and device hardware environments."""
    if not logger.isEnabledFor(logging.INFO):
        return

    affinity_mask = os.sched_getaffinity(0)

    memory = psutil.virtual_memory()

    info = []

    info.append(f"Number of CPUs: {len(affinity_mask)}/{os.cpu_count() or '-'}")
    info.append(f"Memory: {memory.total // (1024 * 1024 * 1024):,}GiB")

    if device is not None and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)

        info.append(f"Device Name: {props.name}")
        info.append(f"Device Memory: {props.total_memory // (1024 * 1024):,}MiB")
        info.append(f"Number of SMs: {props.multi_processor_count}")
        info.append(f"Compute Capability: {props.major}.{props.minor}")

    s = " | ".join(info)

    logger.info(f"Hardware Info - {s}")
