# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from functools import cached_property
from typing import final

import clusterscope
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
    @cached_property
    def _job(self) -> clusterscope.job_info.JobInfo:
        try:
            return clusterscope.get_job()
        except RuntimeError as ex:
            raise OperationalError("`clusterscope.get_job()` has failed.") from ex

    @override
    def set_torch_distributed_env_variables(self) -> None:
        if not self._job.is_slurm_srun():
            raise ClusterNotDetectedError("slurm")

        try:
            self._job.set_torch_distributed_env_from_slurm()
        except RuntimeError as ex:
            raise OperationalError("SLURM job information cannot be retrieved.") from ex

    @override
    def supports_current_cluster(self) -> bool:
        return self._job.is_slurm_srun()

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
