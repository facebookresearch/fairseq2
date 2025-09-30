# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import platform
import socket
from abc import ABC, abstractmethod
from typing import final

import fairseq2n
import psutil
import torch
from rich.pretty import pretty_repr
from typing_extensions import override

import fairseq2
from fairseq2.device import Device
from fairseq2.logging import log
from fairseq2.metrics import format_as_byte_size
from fairseq2.utils.structured import ValueConverter


class _LogHelper(ABC):
    @abstractmethod
    def log_system_info(self) -> None: ...

    @abstractmethod
    def log_software_info(self) -> None: ...

    @abstractmethod
    def log_environment_variables(self) -> None: ...

    @abstractmethod
    def log_config(self, title: str, config: object) -> None: ...


@final
class _StandardLogHelper(_LogHelper):
    def __init__(self, device: Device, value_converter: ValueConverter) -> None:
        self._device = device
        self._value_converter = value_converter

    @override
    def log_system_info(self) -> None:
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

        if self._device.type == "cpu":
            s = "CPU-only"
        elif self._device.type == "cuda":
            props = torch.cuda.get_device_properties(self._device)

            s = (
                f"ID: {self._device} | "
                f"Name: {props.name} | "
                f"Memory: {format_as_byte_size(props.total_memory)} | "
                f"Number of SMs: {props.multi_processor_count} | "
                f"Compute Capability: {props.major}.{props.minor}"
            )
        else:
            s = f"ID: {self._device}"

        log.info("Device - {}", s)

    @override
    def log_software_info(self) -> None:
        if not log.is_enabled_for_info():
            return

        s = f"Python: {platform.python_version()} | PyTorch: {torch.__version__}"

        if torch.cuda.is_available():
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

    @override
    def log_environment_variables(self) -> None:
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

    @override
    def log_config(self, title: str, config: object) -> None:
        unstructured_config = self._value_converter.unstructure(config)

        log.info("{}: {}", title, pretty_repr(unstructured_config, max_width=88))
