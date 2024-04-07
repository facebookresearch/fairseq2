# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import Any, Generator, Optional, Union

import psutil
import torch
from torch.cuda import OutOfMemoryError
from torch.nn import Module

from fairseq2.typing import Device
from fairseq2.utils.dataclass import _dump_dataclass
from fairseq2.utils.logging import LogWriter


@contextmanager
def exception_logger(log: LogWriter) -> Generator[None, None, None]:
    """Log exceptions and CUDA OOM errors raised within the context."""
    try:
        yield
    except OutOfMemoryError:
        s = torch.cuda.memory_summary()

        log.exception("CUDA run out of memory. See the memory stats and the exception details below.\n{}", s)  # fmt: skip

        raise
    except Exception:
        log.exception("Job has failed. See the exception details below.")

        raise


def log_config(config: Any, log: LogWriter, file: Optional[Path] = None) -> None:
    """Log ``config``.

    :param config:
        The config to log. Must be a :class:`~dataclasses.dataclass`.
    :param log:
        The log to write to.
    :param file:
        The output file to write ``config`` in YAML format.
    """
    if file is not None:
        _dump_dataclass(config, file)

    log.info("Config:\n{}", config)


def log_environment_info(
    log: Union[LogWriter, Logger], device: Optional[Device] = None
) -> None:
    """Log information about the software and hardware environments."""
    log_software_info(log, device)
    log_hardware_info(log, device)


# compat
# TODO: Keep only LogWriter
def log_software_info(
    log: Union[LogWriter, Logger], device: Optional[Device] = None
) -> None:
    """Log information about the software environment."""
    if isinstance(log, Logger):
        log = LogWriter(log)

    if not log.is_enabled_for(logging.INFO):
        return

    s = f"PyTorch: {torch.__version__}"

    if device is not None and device.type == "cuda":
        s = f"{s} | CUDA: {torch.version.cuda}"

    s = f"{s} | Intraop Thread Count: {torch.get_num_threads()}"

    log.info("Software Info - {}", s)


def log_hardware_info(
    log: Union[LogWriter, Logger], device: Optional[Device] = None
) -> None:
    """Log information about the host and device hardware environments."""
    if isinstance(log, Logger):
        log = LogWriter(log)

    if not log.is_enabled_for(logging.INFO):
        return

    affinity_mask = os.sched_getaffinity(0)

    memory = psutil.virtual_memory()

    s = (
        f"Number of CPUs: {len(affinity_mask)}/{os.cpu_count() or '-'} | "
        f"Memory: {memory.total // (1024 * 1024 * 1024):,}GiB"
    )

    if device is not None:
        s = f"{s} | Device Name: {device}"

    if device is not None and device.type == "cuda":
        pr = torch.cuda.get_device_properties(device)

        s = (
            f"{s} | "
            f"Device Display Name: {pr.name} | "
            f"Device Memory: {pr.total_memory // (1024 * 1024):,}MiB | "
            f"Number of SMs: {pr.multi_processor_count} | "
            f"Compute Capability: {pr.major}.{pr.minor}"
        )

    log.info("Hardware Info - {}", s)


def log_module(module: Module, log: Union[LogWriter, Logger]) -> None:
    """Log information about ``module`` and its descendants."""
    # compat
    # TODO: move to module scope.
    from fairseq2.nn.utils.module import get_module_size

    if isinstance(log, Logger):
        log = LogWriter(log)

    if not log.is_enabled_for(logging.INFO):
        return

    si = get_module_size(module)

    s = (
        f"Parameter Size: {si.param_size:,} | "
        f"Parameter Size (bytes): {si.param_size_bytes:,} | "
        f"Trainable Parameter Size: {si.trainable_param_size:,} | "
        f"Trainable Parameter Size (bytes): {si.trainable_param_size_bytes:,} | "
        f"Buffer Size: {si.buffer_size:,} | "
        f"Buffer Size (bytes): {si.buffer_size_bytes:,} | "
        f"Total Size: {si.total_size:,} | "
        f"Total Size (bytes): {si.total_size_bytes:,}"
    )

    log.info("Module Info - {} | Layout:\n{}", s, module)
