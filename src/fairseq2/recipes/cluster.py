# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Collection
from random import Random
from typing import final

from typing_extensions import override

from fairseq2.error import AlreadyExistsError
from fairseq2.extensions import run_extensions
from fairseq2.gang import get_rank, get_world_size


@final
class ClusterRegistry:
    _entries: dict[str, ClusterHandler]
    _is_torchrun: bool

    def __init__(self, *, is_torchrun: bool = False) -> None:
        self._entries = {}
        self._is_torchrun = is_torchrun

    def get(self, name: str) -> ClusterHandler:
        if self._is_torchrun or name == "none":
            return _NoneClusterHandler()

        if name == "auto":
            for handler in self._entries.values():
                if handler.supports_current_cluster():
                    return handler

            return _NoneClusterHandler()

        try:
            return self._entries[name]
        except KeyError:
            raise UnknownClusterError(name) from None

    def register(self, name: str, handler: ClusterHandler) -> None:
        if name in self._entries:
            raise AlreadyExistsError(
                f"The registry has already a cluster handler named '{name}'."
            )

        self._entries[name] = handler

    def names(self) -> Collection[str]:
        return self._entries.keys()


class ClusterHandler(ABC):
    @abstractmethod
    def set_torch_distributed_variables(self) -> tuple[int, int]:
        """Set environment variables required to initialize ``torch.distributed``."""

    @abstractmethod
    def supports_current_cluster(self) -> bool:
        """Return ``True`` if this instance supports the current cluster."""


class UnknownClusterError(LookupError):
    cluster: str

    def __init__(self, cluster: str) -> None:
        super().__init__(f"'{cluster}' is not a known cluster.")

        self.cluster = cluster


class ClusterError(Exception):
    cluster: str

    def __init__(self, cluster: str, message: str) -> None:
        super().__init__(message)

        self.cluster = cluster


@final
class SlurmClusterHandler(ClusterHandler):
    _job_id: int | None

    def __init__(self) -> None:
        self._job_id = None

    @override
    def set_torch_distributed_variables(self) -> tuple[int, int]:
        job_id = self._ensure_job_id()

        try:
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["RANK"] = os.environ["SLURM_PROCID"]

            try:
                os.environ["LOCAL_WORLD_SIZE"] = os.environ["SLURM_NTASKS_PER_NODE"]
            except KeyError:
                os.environ["LOCAL_WORLD_SIZE"] = "1"

            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

            os.environ["MASTER_ADDR"] = self._get_master_addr()
            os.environ["MASTER_PORT"] = self._get_master_port(job_id)

            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
        except KeyError as ex:
            raise ClusterError(
                "slurm", "Slurm job environment variables are not set correctly. If you are within an allocated job (i.e. `salloc`), make sure to run with `srun`. If you want to run without Slurm, use `--cluster none`."  # fmt: skip
            ) from ex

        return get_world_size(), get_rank()

    def _ensure_job_id(self) -> int:
        if self._job_id is not None:
            return self._job_id

        try:
            job_id = os.environ["SLURM_JOB_ID"]
        except KeyError:
            raise ClusterError(
                "slurm", "`SLURM_JOB_ID` environment variable does not exist."
            ) from None

        try:
            self._job_id = int(job_id)
        except ValueError as ex:
            raise ClusterError("slurm", "Slurm job ID cannot be parsed.") from ex

        return self._job_id

    @staticmethod
    def _get_master_addr() -> str:
        nodes = os.environ["SLURM_JOB_NODELIST"]

        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodes], capture_output=True, text=True
        )

        if result.returncode == 0:
            if node_list := result.stdout.split("\n"):
                return node_list[0]

        raise ClusterError(
            "slurm", "The hostname or IP address of the Slurm node corresponding to rank 0 cannot be retrieved."  # fmt: skip
        )

    @staticmethod
    def _get_master_port(job_id: int) -> str:
        try:
            return os.environ["MASTER_PORT"]
        except KeyError:
            pass

        return str(Random(job_id).randint(20_000, 60_000))

    @override
    def supports_current_cluster(self) -> bool:
        return "SLURM_JOB_ID" in os.environ


@final
class _NoneClusterHandler(ClusterHandler):
    @override
    def set_torch_distributed_variables(self) -> tuple[int, int]:
        return get_world_size(), get_rank()

    @override
    def supports_current_cluster(self) -> bool:
        return True


def register_clusters(registry: ClusterRegistry) -> None:
    registry.register("slurm", SlurmClusterHandler())

    run_extensions("register_fairseq2_clusters", registry)
