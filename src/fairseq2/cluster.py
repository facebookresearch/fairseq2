# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import final

import clusterscope
from typing_extensions import override

from fairseq2.error import InvalidOperationError
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.utils.env import Environment


def set_torch_distributed_env_variables(cluster: str = "auto") -> str:
    resolver = get_dependency_resolver()

    cluster_resolver = resolver.resolve(_ClusterResolver)

    handler = cluster_resolver.resolve(cluster)

    handler.set_torch_distributed_env_variables()

    return handler.cluster


class _ClusterResolver(ABC):
    @abstractmethod
    def resolve(self, name: str) -> ClusterHandler:
        """
        :raises ClusterNotKnownError:
        """

    @property
    @abstractmethod
    def supported_clusters(self) -> Sequence[str]: ...


@final
class _StandardClusterResolver(_ClusterResolver):
    def __init__(self, env: Environment, handlers: Iterable[ClusterHandler]) -> None:
        names = [h.cluster for h in handlers]

        names.sort()

        self._env = env
        self._handlers = {h.cluster: h for h in handlers}
        self._names = names

    @override
    def resolve(self, name: str) -> ClusterHandler:
        if name == "none":
            return _NoneClusterHandler()

        handler: ClusterHandler | None

        if name == "auto":
            if self._env.has("RANK") and self._env.has("WORLD_SIZE"):
                return _NoneClusterHandler()

            for handler in self._handlers.values():
                if handler.supports_current_cluster():
                    return handler

            return _NoneClusterHandler()

        handler = self._handlers.get(name)
        if handler is None:
            raise LookupError()

        return handler

    @property
    @override
    def supported_clusters(self) -> Sequence[str]:
        return self._names


class ClusterHandler(ABC):
    @abstractmethod
    def set_torch_distributed_env_variables(self) -> None:
        """
        Sets environment variables required to initialize ``torch.distributed``.

        :raises ClusterNotDetectedError:
        :raises RuntimeError:
        """

    @abstractmethod
    def supports_current_cluster(self) -> bool:
        """Returns ``True`` if this handler supports the current cluster."""

    @property
    @abstractmethod
    def cluster(self) -> str: ...


@final
class SlurmHandler(ClusterHandler):
    @cached_property
    def _job(self) -> clusterscope.job_info.JobInfo:
        return clusterscope.get_job()

    @override
    def set_torch_distributed_env_variables(self) -> None:
        if not self._job.is_slurm_srun():
            raise InvalidOperationError("Process is not running on a Slurm cluster.")

        try:
            self._job.set_torch_distributed_env_from_slurm()
        except RuntimeError as ex:
            raise RuntimeError("clusterscope failed.") from ex

    @override
    def supports_current_cluster(self) -> bool:
        return self._job.is_slurm_srun()

    @property
    @override
    def cluster(self) -> str:
        return "Slurm"


@final
class _NoneClusterHandler(ClusterHandler):
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
