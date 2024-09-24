# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from random import Random
from typing import final

from typing_extensions import override

from fairseq2.dependency import DependencyContainer, DependencyResolver


class EnvironmentSetter(ABC):
    """Sets job environment variables."""

    @abstractmethod
    def set_torch_distributed_variables(self) -> None:
        """Set environment variables required to initialize ``torch.distributed``."""

    @abstractmethod
    def supports_current_cluster(self) -> bool:
        """Return ``True`` if this instance supports the current cluster."""

    @property
    @abstractmethod
    def supported_cluster(self) -> str:
        """The cluster type that this instance supports."""


class ClusterNotDetectedError(RuntimeError):
    """Raised when an :class:`EnvironmentSetter` cannot detect its cluster."""


@final
class SlurmEnvironmentSetter(EnvironmentSetter):
    """Sets job environment variables on a Slurm cluster."""

    _job_id: int | None

    def __init__(self) -> None:
        self._job_id = None

    @override
    def set_torch_distributed_variables(self) -> None:
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
            raise RuntimeError(
                "Slurm job environment variables are not correctly set. If you are within an allocated job (i.e. `salloc`), make sure to run with `srun`. If you want to run without Slurm, use `--cluster none`."
            ) from ex

    def _ensure_job_id(self) -> int:
        if self._job_id is not None:
            return self._job_id

        try:
            job_id = os.environ["SLURM_JOB_ID"]
        except KeyError:
            raise ClusterNotDetectedError(
                "Slurm not detected. `SLURM_JOB_ID` environment variable cannot be found."
            ) from None

        try:
            self._job_id = int(job_id)
        except ValueError as ex:
            raise RuntimeError("Slurm job ID cannot be parsed.") from ex

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

        raise RuntimeError(
            "The hostname or IP address of the Slurm node corresponding to rank 0 cannot be retrieved."
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
        try:
            self._ensure_job_id()
        except ClusterNotDetectedError:
            return False

        return True

    @property
    @override
    def supported_cluster(self) -> str:
        return "slurm"


@final
class _NoneEnvironmentSetter(EnvironmentSetter):
    @override
    def set_torch_distributed_variables(self) -> None:
        return

    @override
    def supports_current_cluster(self) -> bool:
        return True

    @property
    @override
    def supported_cluster(self) -> str:
        return "none"


def register_objects(container: DependencyContainer) -> None:
    container.register_factory(EnvironmentSetter, _create_inferred_environment_setter)

    container.register_instance(
        EnvironmentSetter, SlurmEnvironmentSetter(), key="slurm"
    )


def _create_inferred_environment_setter(
    resolver: DependencyResolver,
) -> EnvironmentSetter:
    if "TORCHELASTIC_RUN_ID" in os.environ:  # means we are in `torchrun`.
        return _NoneEnvironmentSetter()

    for _, env_setter in resolver.resolve_all_keyed(EnvironmentSetter):
        if env_setter.supports_current_cluster():
            return env_setter

    return _NoneEnvironmentSetter()
