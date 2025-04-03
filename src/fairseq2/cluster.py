# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import socket
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, MutableMapping
from contextlib import closing
from random import Random
from typing import Any, Dict, final

try:
    import ray  # type: ignore[import-not-found]

    _has_ray = True
except ImportError:
    _has_ray = False

from typing_extensions import override

from fairseq2.logging import log
from fairseq2.registry import Provider
from fairseq2.utils.env import get_rank


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

    @property
    @abstractmethod
    def supported_cluster(self) -> str: ...


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

    @property
    @override
    def supported_cluster(self) -> str:
        return "slurm"


@final
class _NoneClusterHandler(ClusterHandler):
    @override
    def set_torch_distributed_variables(self) -> None:
        pass

    @override
    def supports_current_cluster(self) -> bool:
        return True

    @property
    @override
    def supported_cluster(self) -> str:
        return "none"


class RayCoordinator:
    NAME = "RAY_FAIRSEQ2_COORDINATOR_NAME"
    LEADER_MAX_RETRIES = 30
    LEADER_RETRY_INTERVAL = 1.0

    def __init__(
        self,
        job_id: int,
        world_size: int,
    ):
        self.job_id = job_id
        self.worker_info: Dict[int, Dict[str, Any]] = {}
        self.leader_port = None
        self.ready_workers = 0
        self.world_size = world_size

    def register_worker(
        self, hostname: str, rank: int, free_port: int | None
    ) -> Dict[str, Any]:
        """Register a worker with its placement group ID and GPU ID"""
        self.ready_workers += 1
        info = {
            "hostname": hostname,
            "rank": rank,
            "ready_workers": self.ready_workers,
            "world_size": self.world_size,
            "leader_port": free_port,
        }
        self.worker_info[rank] = info
        return info

    def get_leader_info(self) -> Dict[str, Any] | None:
        if self.ready_workers == self.world_size:
            return self.worker_info[0]
        else:
            return None


@final
class RayClusterHandler(ClusterHandler):
    _env: MutableMapping[str, str]

    def __init__(self, env: MutableMapping[str, str]) -> None:
        self._env = env

    @override
    def set_torch_distributed_variables(self) -> None:
        env = self._env

        rank = get_rank(env)
        hostname = socket.gethostname()

        # Get the coordinator name from environment variable
        coordinator_name = env.get(RayCoordinator.NAME)
        assert coordinator_name
        coordinator = ray.get_actor(*coordinator_name.split(":"))

        free_port = None
        if rank == 0:
            free_port = self.find_free_port()
        worker_info = ray.get(
            coordinator.register_worker.remote(hostname, rank, free_port)
        )

        log.info(f"Worker info: {worker_info}")

        leader = None
        for attempts in range(RayCoordinator.LEADER_MAX_RETRIES):
            leader = ray.get(coordinator.get_leader_info.remote())
            if leader is not None:
                break
            time.sleep(RayCoordinator.LEADER_RETRY_INTERVAL * (1.1**attempts))
        if not leader:
            raise TimeoutError(f"Worker {rank} timed out waiting")

        env["WORLD_SIZE"] = str(worker_info["world_size"])
        env["MASTER_ADDR"] = str(leader["hostname"])
        env["MASTER_PORT"] = str(leader["leader_port"])

    def find_free_port(self) -> int:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
            return int(port)

    @override
    def supports_current_cluster(self) -> bool:
        return _has_ray and "RAY_FAIRSEQ2_COORDINATOR_NAME" in self._env

    @property
    @override
    def supported_cluster(self) -> str:
        return "ray"
