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
from random import Random
from typing import final

import ray
from typing_extensions import override

from fairseq2.logging import log
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


@ray.remote(num_cpus=0)
class RayCoordinator:
    NAME = "RAY_FAIRSEQ2_COORDINATOR_NAME"
    LEADER_MAX_RETRIES = 30
    LEADER_RETRY_INTERVAL = 1.0

    def __init__(self, job_id, num_nodes, gpus_per_node, placement_group_ids):
        self.job_id = job_id
        self.worker_info = {}
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.leader = None
        self.leader_port = None
        self.ready_workers = 0
        self.total_workers = num_nodes * gpus_per_node
        self.placement_group_ids = placement_group_ids

    def register_worker(self, worker_id, hostname, node_id, placement_group_id, gpu_id):
        """Register a worker with its placement group ID and GPU ID"""
        assert (
            placement_group_id in self.placement_group_ids
        ), f"Placement group id {placement_group_id} should in {self.placement_group_ids}"
        assert 0 <= gpu_id <= 7, f"Bad gpu_id {gpu_id}"
        node_index = self.placement_group_ids.index(placement_group_id)
        global_rank = node_index * self.gpus_per_node + gpu_id
        info = {
            "worker_id": worker_id,
            "hostname": hostname,
            "node_id": node_id,
            "local_rank": gpu_id,
            "placement_group_id": placement_group_id,
            "global_rank": global_rank,
            "world_size": self.total_workers,
            "local_world_size": self.gpus_per_node,
        }
        self.worker_info[worker_id] = info

        self.ready_workers += 1

        is_leader = node_index == 0 and gpu_id == 0

        if is_leader:
            self.leader_port = str(Random(self.job_id).randint(20_000, 60_000))
            self.leader = info

        return {
            **self.worker_info[worker_id],
            "is_leader": is_leader,
            "ready_workers": self.ready_workers,
            "total_workers": self.total_workers,
        }

    def get_leader_info(self):
        if self.leader and self.ready_workers == self.total_workers:
            result = self.leader.copy()
            result["leader_port"] = self.leader_port
            return result
        else:
            return None


@final
class RayClusterHandler(ClusterHandler):
    _job_id: int | None
    _env: MutableMapping[str, str]

    def __init__(self, env: MutableMapping[str, str]) -> None:
        self._job_id = None
        self._env = env

    @override
    def set_torch_distributed_variables(self) -> None:
        job_id = self._ensure_job_id()

        env = self._env

        ray_context = ray.get_runtime_context()
        self.actor_id = ray_context.get_actor_id()
        self.worker_id = str(self.actor_id)
        self.node_id = ray_context.get_node_id()
        self.hostname = socket.gethostname()
        self.placement_group_id = ray_context.get_placement_group_id()

        # Get the coordinator name from environment variable
        coordinator_name = env.get(RayCoordinator.NAME)
        assert coordinator_name
        self.coordinator = ray.get_actor(*coordinator_name.split(":"))

        accelerator_ids = ray_context.get_accelerator_ids()
        if (
            not accelerator_ids
            or "GPU" not in accelerator_ids
            or not accelerator_ids["GPU"]
        ):
            raise ValueError("No GPU accelerator ID available")
        self.gpu_id = int(accelerator_ids["GPU"][0])

        assert self.gpu_id == int(env["CUDA_VISIBLE_DEVICES"])

        self.worker_info = ray.get(
            self.coordinator.register_worker.remote(
                self.worker_id,
                self.hostname,
                self.node_id,
                self.placement_group_id,
                self.gpu_id,
            )
        )

        log.info(f"Worker info: {self.worker_info}")

        leader = None
        for attempts in range(RayCoordinator.LEADER_MAX_RETRIES):
            leader = ray.get(self.coordinator.get_leader_info.remote())
            if leader:
                break
            time.sleep(RayCoordinator.LEADER_RETRY_INTERVAL * (1.1**attempts))
        if not leader:
            raise TimeoutError(
                f"Worker {self.worker_id} (rank {self.global_rank}) timed out waiting"
            )

        env["WORLD_SIZE"] = str(self.worker_info["world_size"])
        env["RANK"] = str(self.worker_info["global_rank"])
        env["LOCAL_WORLD_SIZE"] = str(self.worker_info["local_world_size"])
        env["LOCAL_RANK"] = str(self.worker_info["local_rank"])

        env["MASTER_ADDR"] = str(leader["hostname"])
        env["MASTER_PORT"] = str(leader["leader_port"])

        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

    def _ensure_job_id(self) -> int:
        if self._job_id is not None:
            return self._job_id

        try:
            job_id = self._env["RAY_JOB_ID"]
        except KeyError:
            raise ClusterError(
                "ray", "`RAY_JOB_ID` environment variable does not exist."
            ) from None

        try:
            self._job_id = int(job_id)
        except ValueError as ex:
            raise ClusterError("ray", "Ray job ID cannot be parsed.") from ex

        return self._job_id

    @override
    def supports_current_cluster(self) -> bool:
        return "RAY_FAIRSEQ2_COORDINATOR_NAME" in self._env

    @property
    @override
    def supported_cluster(self) -> str:
        return "ray"
