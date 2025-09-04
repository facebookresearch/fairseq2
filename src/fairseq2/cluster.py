# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from random import Random
from typing import final

from typing_extensions import override

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
