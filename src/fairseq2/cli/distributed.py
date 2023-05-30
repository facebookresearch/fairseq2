import logging
import math
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

import submitit
import torch
import torchtnt.utils.distributed

log = logging.getLogger(__name__)


class Env(NamedTuple):
    """Represents the distributed environment we are currently running in."""

    world_size: int
    """Total number of worker process working together"""

    global_rank: int
    """Unique id of this worker. Workers are numbered from 0 to ``world_size - 1``"""

    device: torch.device
    """Cuda device this worker should use."""


def env(device: Optional[torch.device] = None) -> Env:
    if device is None:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    return Env(
        global_rank=torchtnt.utils.distributed.get_global_rank(),
        world_size=torchtnt.utils.distributed.get_world_size(),
        device=device,
    )


def distributed_init(
    workdir: Path,
    partition: str,
    num_gpus: int,
    timeout: timedelta = timedelta(days=7),
    cpu_per_gpu: int = 4,
    num_gpu_per_node: int = 8,
    qos: Optional[str] = None,
    slurm_args: Dict[str, Any] = {},
) -> Env:
    """
    Makes sure that the current python program is run with the requested number of GPU.
    If not, it will spawn one process per GPU, typically using Slurm.

    workdir: the working dir. That's where the log will appear.
    partition: the partition to use to run on Slurm.
        There is also two special partitions:
            * "debug" will run the program normally using the first GPU (cuda:0)
            * "local" will run the program locally, spawning one subprocess per GPU.
    num_gpus: number of GPU to use.
    timeout: duration after which we should stop the training.
    cpu_per_gpu: number of cpu per gpu we should use.
    num_gpu_per_node: decide how much GPU we should use per node.
        Most clusters have 8 GPUs per node, so that's the default.
    qos: --qos flag for SLURM
    one_file: if true, the caller guarantees that the current python script is movable.
        We will copy the script into the workdir and run it from there.
        This allows to continue editing the original script file, while the Slurm job run
        with its own copy.
    slurm_args: extra arguments for Slurm `sbatch` command line.

    Returns: either don't return if we need to spawn jobs/processes, or return an Env
    describing the rank and world size of the current process.
    """
    # TODO: should I replace submitit by torchx ? https://github.com/fairinternal/fairseq2/issues/359
    import __main__

    # Copy the main script to the workdir.
    main_py = Path(__main__.__file__)

    # Check if we are already a distributed worker
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    if world_size < 0:
        world_size = int(os.environ.get("SLURM_NTASKS", -1))
    if world_size < 0:
        world_size = int(os.environ.get("SUBMITIT_LOCAL_NTASKS", -1))

    if world_size == -1 and partition != "debug":
        timeout_min = int(timeout.total_seconds() // 60)
        max_num_timeout = 3
        if submitit.AutoExecutor.which() == "slurm":
            timeout_min = min(timeout_min, 24 * 60)
            max_num_timeout = max(max_num_timeout, math.ceil(timeout_min / 24 * 60))

        # We aren't running in distributed mode, let submit a job to do so.
        ex = submitit.AutoExecutor(
            folder=workdir, slurm_max_num_timeout=max_num_timeout
        )
        ex.update_parameters(
            name=main_py.stem,
            nodes=max(num_gpus // num_gpu_per_node, 1),
            gpus_per_node=min(num_gpus, num_gpu_per_node),
            tasks_per_node=min(num_gpus, num_gpu_per_node) or 1,
            cpus_per_task=cpu_per_gpu,
            timeout_min=timeout_min,
            slurm_partition=partition,
            slurm_qos=qos,
            slurm_additional_parameters=slurm_args,
        )

        # Note: we aren't changing cwd, I think that's generally less surprising,
        # but also more error prone.
        job = ex.submit(RunScript(main_py))
        log.info(
            f"Scheduled training job: {job.job_id}.\nLogs will be at {job.paths.stderr}.\nYou can exit this process with ctrl+c, Slurm will take care of running the training job."
        )
        # TODO: poll log file
        job_state = job.state
        try:
            while not job.done():
                new_job_state = job.state
                if new_job_state != job_state:
                    print(job)
                    job_state = new_job_state
                else:
                    time.sleep(10)
        except KeyboardInterrupt:
            sys.exit(0)

        job.wait()
        exc = job.exception()
        if exc is not None:
            stderr = parse_submitit_stderr(job.stderr() or "")
            log.error(f"{job} triggered an exception: {stderr}")
            log.error(f"Full log at {job.paths.stderr}")
            sys.exit(1)

        res = job.result()
        log.info(f"Training is done ! Result: {res}")
        sys.exit(0)

    if world_size == -1 and partition == "debug":
        if num_gpus <= 1:
            device = torch.device("cuda:0") if num_gpus else torch.device("cpu")
            log.info(f"Starting local training {sys.executable} on device {device}.")
            return Env(world_size=1, global_rank=0, device=device)

        if num_gpus > 1:
            _torchx_local(main_py, workdir, num_gpus)
            sys.exit(0)

    try:
        # Handle the case where we are launched from submitit
        # ie with fairseq2 train --num_gpus=8
        # WORLD_SIZE, LOCAL_RANK, etc ... aren't set yet,
        # TorchDistributedEnvironment will take care of that by parsing the SLURM
        # env variables and converting them to torch env variables.
        env = submitit.helpers.TorchDistributedEnvironment()
        if env.world_size == world_size:
            env.export(set_cuda_visible_devices=True)
            os.environ["GROUP_RANK"] = "0"
            os.environ.update(os.environ)
    except RuntimeError:
        pass

    # Following https://pytorch.org/docs/stable/generated/torch.cuda.set_device.html
    # we try to set CUDA_VISIBLE_DEVICES if this wasn't done before we start.
    # TODO: should we set CUDA_VISIBLE_DEVICES or just return a unique "device" per worker ?
    visible_devices = [x for x in os.getenv("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    n_devices = len(visible_devices)
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", 0))

    if n_devices > 1 and n_devices == local_world_size:
        log.warning(
            f"{n_devices} devices and {local_world_size} workers. Assigning 1 GPU per worker."
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[local_rank]
    elif n_devices == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    elif n_devices != local_world_size:
        log.warning(
            f"{n_devices} devices and {local_world_size} workers. Each worker will see all GPUs."
        )

    device = torch.device("cuda:0")
    log.info(
        f"Starting distributed worker {sys.executable} on device {device}.\n"
        + "".join(
            f"{k}: {os.environ.get(k)}\n"
            for k in [
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "CUDA_VISIBLE_DEVICES",
            ]
        )
    )

    torch.distributed.init_process_group(backend="nccl")
    return Env(
        world_size=int(os.environ["WORLD_SIZE"]),
        global_rank=int(os.environ["RANK"]),
        device=device,
    )


def parse_submitit_stderr(stderr: str, nlines: int = 10) -> str:
    stderr = stderr.rsplit("Submitted job triggered an exception\n", 1)[0]
    traces = stderr.split("Traceback (most recent call last):\n", 1)
    stderr, trace = traces if len(traces) == 2 else (stderr, "")
    last_stderr_lines = stderr.rsplit(os.linesep, nlines)
    last_stderr_lines.append(trace)
    return "\n".join(last_stderr_lines)


def _torchx_local(main_py: Path, workdir: Path, num_gpus: int) -> None:
    """Uses torchx to start a local multi-gpu job.

    Note: this isn't supported by submitit, and torchx local runner works well.
    """
    subprocess.run(
        [
            "torchx",
            "run",
            "--scheduler=local_cwd",
            "--scheduler_args",
            f"log_dir={workdir}",
            "--log",
            "--workspace=''",
            "dist.ddp",
            "--cpu=4",
            f"--gpu={num_gpus}",
            "-j",
            f"1x{num_gpus}",
            "--script",
            str(main_py),
        ]
        + sys.argv[1:]
    )


class RunScript(submitit.helpers.Checkpointable):
    """Run the given python main script. Tell submitit it's allowed to resubmit the job in case of timeout."""

    def __init__(self, main_py: Path):
        self.executable = sys.executable
        self.argv = sys.argv[1:]
        self.main_py = main_py

    def __call__(self) -> None:
        # Note: spawning this subprocess is not free,
        # in particular we lost the ability to return an exception or result to the caller.
        # But this avoid pickling issues.
        # pickle doesn't like stuff defined in the program entry point.
        # TODO: try runpy.run_path this should be equivalent but happens in the same process
        main_py = self.main_py
        subprocess.run(
            [sys.executable, str(main_py)] + self.argv,
            check=True,
        )
