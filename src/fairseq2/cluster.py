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
from contextlib import closing
from collections.abc import Collection, Iterable
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
from fairseq2.error import OperationalError
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.utils.env import Environment


def set_torch_distributed_env_variables(cluster: str = "auto") -> str:
    resolver = get_dependency_resolver()

    cluster_resolver = resolver.resolve(ClusterResolver)

    handler = cluster_resolver.resolve(cluster)

    handler.set_torch_distributed_env_variables()

    return handler.cluster


class ClusterResolver(ABC):
    @abstractmethod
    def resolve(self, name: str) -> ClusterHandler: ...


@final
class StandardClusterResolver(ClusterResolver):
    def __init__(self, env: Environment, handlers: Iterable[ClusterHandler]) -> None:
        self._env = env
        self._handlers = {h.cluster: h for h in handlers}

    def resolve(self, name: str) -> ClusterHandler:
        if name == "none":
            return NoneClusterHandler()

        handler: ClusterHandler | None

        if name == "auto":
            if self._env.has("RANK") and self._env.has("WORLD_SIZE"):
                return NoneClusterHandler()

            for handler in self._handlers.values():
                if handler.supports_current_cluster():
                    return handler

            return NoneClusterHandler()

        handler = self._handlers.get(name)
        if handler is None:
            raise ClusterNotKnownError(name, self._handlers.keys())

        return handler


class ClusterNotKnownError(Exception):
    def __init__(self, cluster: str, known_clusters: Collection[str]) -> None:
        super().__init__(f"{cluster} is not a known cluster.")

        self.cluster = cluster
        self.known_clusters = known_clusters


class ClusterHandler(ABC):
    @abstractmethod
    def set_torch_distributed_env_variables(self) -> None:
        """Set environment variables required to initialize ``torch.distributed``."""

    @abstractmethod
    def supports_current_cluster(self) -> bool:
        """Return ``True`` if this instance supports the current cluster."""

    @property
    @abstractmethod
    def cluster(self) -> str: ...


class ClusterNotDetectedError(Exception):
    def __init__(self, cluster: str) -> None:
        super().__init__(f"Process is not running on a {cluster} cluster.")

        self.cluster = cluster


@final
class SlurmHandler(ClusterHandler):
    def __init__(self, env: Environment) -> None:
        self._job_id: int | None = None
        self._env = env

    @override
    def set_torch_distributed_env_variables(self) -> None:
        env = self._env

        if not env.has("SLURM_PROCID"):
            raise ClusterNotDetectedError("slurm")

        job_id = self._ensure_job_id()

        try:
            env.set("WORLD_SIZE", env.get("SLURM_NTASKS"))
            env.set("RANK", env.get("SLURM_PROCID"))

            try:
                env.set("LOCAL_WORLD_SIZE", env.get("SLURM_NTASKS_PER_NODE"))
            except KeyError:
                env.set("LOCAL_WORLD_SIZE", "1")

            env.set("LOCAL_RANK", env.get("SLURM_LOCALID"))

            env.set("MASTER_ADDR", self._get_master_addr())
            env.set("MASTER_PORT", self._get_master_port(job_id))

            env.set("CUDA_VISIBLE_DEVICES", env.get("SLURM_LOCALID"))
        except KeyError as ex:
            raise OperationalError(
                "Slurm job environment variables are not set correctly."
            ) from ex

    def _ensure_job_id(self) -> int:
        if self._job_id is not None:
            return self._job_id

        try:
            job_id = self._env.get("SLURM_JOB_ID")
        except KeyError:
            raise OperationalError(
                "SLURM_JOB_ID environment variable does not exist."
            ) from None

        try:
            self._job_id = int(job_id)
        except ValueError:
            raise OperationalError("Slurm job ID cannot be parsed.") from None

        return self._job_id

    def _get_master_addr(self) -> str:
        nodes = self._env.get("SLURM_JOB_NODELIST")

        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodes], capture_output=True, text=True
        )

        if result.returncode == 0:
            if node_list := result.stdout.split("\n"):
                return node_list[0]

        raise OperationalError(
            "Hostname or IP address of the Slurm node corresponding to rank 0 cannot be retrieved."  # fmt: skip
        )

    def _get_master_port(self, job_id: int) -> str:
        port = self._env.maybe_get("MASTER_PORT")
        if port is not None:
            return port

        return str(Random(job_id).randint(20_000, 60_000))

    @override
    def supports_current_cluster(self) -> bool:
        return self._env.has("SLURM_PROCID")

    @property
    @override
    def cluster(self) -> str:
        return "slurm"


@final
class NoneClusterHandler(ClusterHandler):
    @override
    def set_torch_distributed_env_variables(self) -> None:
        pass

    @override
    def supports_current_cluster(self) -> bool:
        return True

    @property
    @override
    def cluster(self) -> str:
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
