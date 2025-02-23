# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import platform
import socket

import fairseq2n
import psutil
import torch
from rich.pretty import pretty_repr
from torch.nn import Module

import fairseq2
from fairseq2.gang import Gangs
from fairseq2.logging import LogWriter
from fairseq2.metrics import format_as_byte_size
from fairseq2.nn.utils.module import get_module_size
from fairseq2.typing import Device


def log_config(log: LogWriter, title: str, config: object) -> None:
    log.info("{}:\n{}", title, pretty_repr(config, max_width=88))


def log_environment_info(log: LogWriter, device: Device) -> None:
    """Log information about the host system and the installed software."""
    log_system_info(log, device)

    log_software_info(log, device)

    log_environment_variables(log)


def log_system_info(log: LogWriter, device: Device) -> None:
    """Log information about the host system."""
    if not log.is_enabled_for_info():
        return

    def read_dist_name() -> str | None:
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

    try:
        affinity_mask = os.sched_getaffinity(0)
    except AttributeError:  # Python on macOS does not have `sched_getaffinity`.
        if num_cpus is None:
            affinity_mask = None
        else:
            affinity_mask = set(range(num_cpus))

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

    s = f"Name: {socket.gethostname()} | Platform: {platform.platform()}"

    if dist_name:
        s = f"{s} | Linux Distribution: {dist_name}"

    s = (
        f"{s} | "
        f"Number of CPUs: {cpu_info} | "
        f"Memory: {format_as_byte_size(memory.total)}"
    )

    log.info("Host - {}", s)

    if device.type == "cpu":
        s = "CPU-only"
    elif device.type == "cuda":
        props = torch.cuda.get_device_properties(device)

        s = (
            f"ID: {device} | "
            f"Name: {props.name} | "
            f"Memory: {format_as_byte_size(props.total_memory)} | "
            f"Number of SMs: {props.multi_processor_count} | "
            f"Compute Capability: {props.major}.{props.minor}"
        )
    else:
        s = f"ID: {device}"

    log.info("Device - {}", s)


def log_software_info(log: LogWriter, device: Device) -> None:
    """Log information about the installed software."""
    if not log.is_enabled_for_info():
        return

    s = f"Python: {platform.python_version()} | PyTorch: {torch.__version__}"

    if device.type == "cuda":
        s = (
            f"{s} | "
            f"CUDA: {torch.version.cuda} | "
            f"NCCL: {'.'.join((str(v) for v in torch.cuda.nccl.version()))}"
        )

    s = f"{s} | fairseq2: {fairseq2.__version__} | fairseq2n: {fairseq2n.__version__}"

    for venv_type, venv_env in [("Conda", "CONDA_PREFIX"), ("venv", "VIRTUAL_ENV")]:
        if venv_path := os.environ.get(venv_env):
            s = f"{s} | Python Environment: {venv_type} ({venv_path})"

    log.info("Software - {}", s)

    s = (
        f"Process ID: {os.getpid()} | "
        f"PyTorch Intraop Thread Count: {torch.get_num_threads()}"
    )

    log.info("Runtime Environment - {}", s)


def log_environment_variables(log: LogWriter) -> None:
    """Log the environment variables."""
    if not log.is_enabled_for_info():
        return

    kv = []

    skip_list = {"PS1", "LS_COLORS", "GREP_COLORS", "GCC_COLORS"}

    for k, v in os.environ.items():
        if k.startswith("BASH_FUNC") or k in skip_list:
            continue

        kv.append(f"{k}={v}")

    log.info("Environment Variables - {}", ", ".join(kv))


def log_model(log: LogWriter, model: Module, gangs: Gangs) -> None:
    """Log information about ``model``."""
    if not log.is_enabled_for_info():
        return

    si = get_module_size(model)

    s = (
        f"Parameter Size: {si.param_size:,} | "
        f"Parameter Size (bytes): {format_as_byte_size(si.param_size_bytes)} | "
        f"Trainable Parameter Size: {si.trainable_param_size:,} | "
        f"Trainable Parameter Size (bytes): {format_as_byte_size(si.trainable_param_size_bytes)} | "
        f"Buffer Size: {si.buffer_size:,} | "
        f"Buffer Size (bytes): {format_as_byte_size(si.buffer_size_bytes)} | "
        f"Total Size: {si.total_size:,} | "
        f"Total Size (bytes): {format_as_byte_size(si.total_size_bytes)}"
    )

    log.info("Model (rank {}) - {} | Layout:\n{}", gangs.root.rank, s, model)
