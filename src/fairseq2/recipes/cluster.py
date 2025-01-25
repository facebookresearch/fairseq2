# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, MutableMapping
from random import Random
from typing import final

from typing_extensions import override

from fairseq2.registry import Provider


@final
class ClusterResolver:
    _handlers: Provider[ClusterHandler]

    def __init__(
        self, handlers: Provider[ClusterHandler], env: Mapping[str, str]
    ) -> None:
        self._handlers = handlers

        self._is_torchrun = "TORCHELASTIC_RUN_ID" in env

    def get(self, name: str) -> ClusterHandler:
        if self._is_torchrun or name == "none":
            return _NoneClusterHandler()

        if name == "auto":
            for _, handler in self._handlers.get_all():
                if handler.supports_current_cluster():
                    return handler

            return _NoneClusterHandler()

        try:
            return self._handlers.get(name)
        except LookupError:
            raise UnknownClusterError(name, self.supported_clusters()) from None

    def supported_clusters(self) -> Collection[str]:
        return [str(key) for key, _ in self._handlers.get_all()]


class UnknownClusterError(Exception):
    cluster: str
    supported_clusters: Collection[str]

    def __init__(self, cluster: str, supported_clusters: Collection[str]) -> None:
        super().__init__(f"'{cluster}' is not a known cluster.")

        self.cluster = cluster
        self.supported_clusters = supported_clusters


class ClusterHandler(ABC):
    @abstractmethod
    def set_torch_distributed_variables(self) -> None:
        """Set environment variables required to initialize ``torch.distributed``."""

    @abstractmethod
    def supports_current_cluster(self) -> bool:
        """Return ``True`` if this instance supports the current cluster."""


class ClusterError(Exception):
    cluster: str

    def __init__(self, cluster: str, message: str) -> None:
        super().__init__(message)

        self.cluster = cluster


@final
class SlurmClusterHandler(ClusterHandler):
    _job_id: int | None
    _env: MutableMapping[str, str]

    def __init__(self, env: MutableMapping[str, str]) -> None:
        self._job_id = None
        self._env = env

    @override
    def set_torch_distributed_variables(self) -> None:
        job_id = self._ensure_job_id()

        env = self._env

        try:
            env["WORLD_SIZE"] = env["SLURM_NTASKS"]
            env["RANK"] = env["SLURM_PROCID"]

            try:
                env["LOCAL_WORLD_SIZE"] = env["SLURM_NTASKS_PER_NODE"]
            except KeyError:
                env["LOCAL_WORLD_SIZE"] = "1"

            env["LOCAL_RANK"] = env["SLURM_LOCALID"]

            env["MASTER_ADDR"] = self._get_master_addr()
            env["MASTER_PORT"] = self._get_master_port(job_id)

            env["CUDA_VISIBLE_DEVICES"] = env["SLURM_LOCALID"]
        except KeyError as ex:
            raise ClusterError(
                "slurm", "Slurm job environment variables are not set correctly."
            ) from ex

    def _ensure_job_id(self) -> int:
        if self._job_id is not None:
            return self._job_id

        try:
            job_id = self._env["SLURM_JOB_ID"]
        except KeyError:
            raise ClusterError(
                "slurm", "`SLURM_JOB_ID` environment variable does not exist."
            ) from None

        try:
            self._job_id = int(job_id)
        except ValueError as ex:
            raise ClusterError("slurm", "Slurm job ID cannot be parsed.") from ex

        return self._job_id

    def _get_master_addr(self) -> str:
        nodes = self._env["SLURM_JOB_NODELIST"]

        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodes], capture_output=True, text=True
        )

        if result.returncode == 0:
            if node_list := result.stdout.split("\n"):
                return node_list[0]

        raise ClusterError(
            "slurm", "The hostname or IP address of the Slurm node corresponding to rank 0 cannot be retrieved."  # fmt: skip
        )

    def _get_master_port(self, job_id: int) -> str:
        try:
            return self._env["MASTER_PORT"]
        except KeyError:
            pass

        return str(Random(job_id).randint(20_000, 60_000))

    @override
    def supports_current_cluster(self) -> bool:
        return "SLURM_JOB_ID" in self._env


@final
class _NoneClusterHandler(ClusterHandler):
    @override
    def set_torch_distributed_variables(self) -> None:
        pass

    @override
    def supports_current_cluster(self) -> bool:
        return True
