# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Literal, Sequence, TypeVar

from submitit import AutoExecutor
from submitit.helpers import TorchDistributedEnvironment, monitor_jobs
from typing_extensions import TypeAlias

JobConfigT = TypeVar("JobConfigT")

JobEntryPoint: TypeAlias = Callable[[JobConfigT], None]


@dataclass
class ClusterConfig:
    cluster: Literal["slurm", "local"]
    parallelism: int
    partition: str
    num_nodes: int
    num_gpus_per_node: int
    cpus_per_task: int
    log_dir: Path
    timeout: timedelta


class Cluster:
    _executor: AutoExecutor

    def __init__(self, config: ClusterConfig) -> None:
        self._executor = AutoExecutor(folder=config.log_dir, cluster=config.cluster)

        if "FAIR_ENV_CLUSTER" in os.environ:
            cluster_args = {
                "slurm_partition": config.partition,
                "slurm_constraint": "volta32gb",
            }
        else:
            cluster_args = {
                "slurm_account": "mms",
                "slurm_qos": config.partition,
            }

        self._executor.update_parameters(
            slurm_array_parallelism=config.parallelism,
            nodes=config.num_nodes,
            cpus_per_task=config.cpus_per_task,
            tasks_per_node=config.num_gpus_per_node,
            gpus_per_node=config.num_gpus_per_node,
            timeout_min=int(config.timeout.total_seconds() // 60),
            **cluster_args
        )

    def run_job(
        self,
        entry_point: JobEntryPoint[JobConfigT],
        config: JobConfigT,
        output_dir: Path,
    ) -> None:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

        job = self._executor.submit(_wrap_entry_point(entry_point), config, output_dir)

        monitor_jobs([job])

    def run_job_array(
        self,
        entry_point: JobEntryPoint[JobConfigT],
        configs: Sequence[JobConfigT],
        output_dirs: Sequence[Path],
    ) -> None:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

        jobs = self._executor.map_array(
            _wrap_entry_point(entry_point), configs, output_dirs
        )

        monitor_jobs(jobs)


def _wrap_entry_point(
    entry_point: JobEntryPoint[JobConfigT],
) -> JobEntryPoint[JobConfigT]:
    def _wrapped_entry_point(config: JobConfigT, output_dir: Path) -> None:
        TorchDistributedEnvironment().export(set_cuda_visible_devices=True)

        entry_point(config, output_dir)

    return _wrapped_entry_point
