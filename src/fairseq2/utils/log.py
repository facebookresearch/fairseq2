# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import platform
import socket
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import Any, Generator, Optional, Union

import fairseq2n
import psutil
import torch
from torch.cuda import OutOfMemoryError
from torch.nn import Module

import fairseq2
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

        log.exception("CUDA run out of memory. See memory stats and exception details below.\n{}", s)  # fmt: skip

        raise
    except Exception:
        log.exception("Job has failed. See exception details below.")

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

    log.info("Job Config:\n{}", config)


# compat
# TODO: Keep only LogWriter
def log_environment_info(
    log: Union[LogWriter, Logger], device: Optional[Device] = None
) -> None:
    """Log information about the host system and the installed software."""
    if isinstance(log, Logger):
        log = LogWriter(log)

    log_system_info(log, device)

    log_software_info(log, device)


def log_system_info(log: LogWriter, device: Optional[Device] = None) -> None:
    """Log information about the host system."""
    if not log.is_enabled_for(logging.INFO):
        return

    def read_dist_name() -> Optional[str]:
        try:
            fp = open("/etc/os-release")
        except OSError:
            return None

        try:
            for line in fp:
                if line.startswith("PRETTY_NAME"):
                    splits = line.rstrip().split("=", maxsplit=1)
                    if len(splits) != 2:
                        break

                    name = splits[1].strip()

                    # Unquote
                    if len(name) >= 2 and name[0] == '"' and name[-1] == '"':
                        name = name[1:-1]

                    return name
        except OSError:
            pass
        finally:
            fp.close()

        return None

    dist_name = read_dist_name()

    num_cpus = os.cpu_count()

    affinity_mask = os.sched_getaffinity(0)

    if num_cpus is None or affinity_mask is None:
        cpu_info = "-"
    else:
        cpu_info = f"{len(affinity_mask)}/{num_cpus}"

        if len(affinity_mask) != num_cpus:
            available_cpus = list(affinity_mask)

            available_cpus.sort()

            cpu_ranges = []

            range_b = available_cpus[0]
            range_e = available_cpus[0]

            for cpu in available_cpus[1:]:
                if cpu == range_e + 1:
                    range_e = cpu

                    continue

                cpu_ranges.append((range_b, range_e))

                range_b = cpu
                range_e = cpu

            cpu_ranges.append((range_b, range_e))

            cpu_range_strs = []

            for range_b, range_e in cpu_ranges:
                if range_b == range_e:
                    cpu_range_strs.append(f"{range_b}")
                else:
                    cpu_range_strs.append(f"{range_b}-{range_e}")

            cpu_info = f"{cpu_info} ({','.join(cpu_range_strs)})"

    memory = psutil.virtual_memory()

    s = f"Name: {socket.getfqdn()} | Platform: {platform.platform()}"

    if dist_name:
        s = f"{s} | Linux Distribution: {dist_name}"

    s = (
        f"{s} | "
        f"Number of CPUs: {cpu_info} | "
        f"Memory: {memory.total // (1024 * 1024 * 1024):,}GiB"
    )

    log.info("Host - {}", s)

    if device is None:
        return

    if device.type == "cpu":
        s = "CPU-only"
    elif device.type == "cuda":
        pr = torch.cuda.get_device_properties(device)

        s = (
            f"ID: {device} | "
            f"Name: {pr.name} | "
            f"Memory: {pr.total_memory // (1024 * 1024):,}MiB | "
            f"Number of SMs: {pr.multi_processor_count} | "
            f"Compute Capability: {pr.major}.{pr.minor}"
        )
    else:
        s = f"ID: {device}"

    log.info("Device - {}", s)


def log_software_info(log: LogWriter, device: Optional[Device] = None) -> None:
    """Log information about the installed software."""
    if not log.is_enabled_for(logging.INFO):
        return

    s = f"Python: {platform.python_version()} | PyTorch: {torch.__version__}"

    if device is not None and device.type == "cuda":
        s = (
            f"{s} | "
            f"CUDA: {torch.version.cuda} | "
            f"NCCL: {'.'.join((str(v) for v in torch.cuda.nccl.version()))}"
        )

    s = (
        f"{s} | "
        f"fairseq2: {fairseq2.__version__} | "
        f"fairseq2n: {fairseq2n.__version__}"
    )

    for venv_type, venv_env in [("Conda", "CONDA_PREFIX"), ("venv", "VIRTUAL_ENV")]:
        if venv_path := os.getenv(venv_env):
            s = f"{s} | Python Environment: {venv_type} ({venv_path})"

    log.info("Software - {}", s)

    s = (
        f"Process ID: {os.getpid()} | "
        f"PyTorch Intraop Thread Count: {torch.get_num_threads()}"
    )

    log.info("Runtime Environment - {}", s)


def log_model(model: Module, log: LogWriter) -> None:
    """Log information about ``model``."""
    # compat
    # TODO: move to module scope.
    from fairseq2.nn.utils.module import get_module_size

    if not log.is_enabled_for(logging.INFO):
        return

    si = get_module_size(model)

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

    log.info("Model - {} | Layout:\n{}", s, model)
