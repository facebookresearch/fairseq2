import logging
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

from fairseq2.typing import Device

log = logging.getLogger(__name__)


class Env(NamedTuple):
    workdir: Path
    global_rank: int
    world_size: int
    device: Device


def env(workdir: Optional[Path] = None, device: Optional[Device] = None) -> Env:
    return Env(
        workdir or Path.cwd(),
        global_rank=torchtnt.utils.distributed.get_global_rank(),
        world_size=torchtnt.utils.distributed.get_world_size(),
        device=device or Device("cpu"),
    )


def init(
    workdir: Path,
    partition: str,
    num_gpus: int,
    timeout: timedelta = timedelta(days=1),
    cpu_per_gpu: int = 4,
    num_gpu_per_node: int = 8,
    qos: Optional[str] = None,
    one_file: bool = False,
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
    # Check if we are already a distributed worker
    world_size = int(os.environ.get("SLURM_NTASKS", -1))
    if world_size < 0:
        world_size = int(os.environ.get("SUBMITIT_LOCAL_NTASKS", -1))
    if world_size < 0:
        world_size = int(os.environ.get("WORLD_SIZE", -1))

    if world_size == -1 and partition != "debug":
        # Copy the main script to the workdir.
        import __main__

        # TODO: rethink this, now "train_mt.py" isn't __main__ anymore
        main_py = Path(__main__.__file__)
        (workdir / main_py.name).write_bytes(main_py.read_bytes())
        # We aren't running in distributed mode, let submit a job to do so.
        ex = submitit.AutoExecutor(
            folder=workdir, cluster="local" if partition == "local" else None
        )
        ex.update_parameters(
            name=main_py.stem,
            nodes=max(num_gpus // num_gpu_per_node, 1),
            gpus_per_node=min(num_gpus, num_gpu_per_node),
            tasks_per_node=min(num_gpus, num_gpu_per_node),
            cpus_per_task=cpu_per_gpu,
            timeout_min=int(timeout.total_seconds() // 60),
            slurm_partition=partition,
            slurm_qos=qos,
            slurm_additional_parameters=slurm_args,
        )

        # Note: spawning this subprocess is not free,
        # in particular we lost the ability to return an exception or result to the caller.
        # But this avoid pickling issues.
        # pickle doesn't like stuff defined in the program entry point.
        # TODO: try runpy.run_path this should be equivalent but happens in the same process
        if one_file:
            # User guarantees to have a one file experience, it's safe to change dir
            job = ex.submit(
                subprocess.run,
                [sys.executable, main_py.name] + sys.argv[1:],
                cwd=workdir,
                check=True,
            )
        else:
            # Else continue from current dir, but the job will pickup the state of the code when it starts
            job = ex.submit(
                subprocess.run,
                [sys.executable, str(main_py)] + sys.argv[1:],
                check=True,
            )
        log.info(
            f"Scheduled training job: {job.job_id}.\nLogs will be at {job.paths.stderr}.\nYou can exit this process with ctrl+c, Slurm will take care of running the training job."
        )
        # TODO: silence keyboard interrupt here
        job_state = job.state
        while not job.done():
            new_job_state = job.state
            if new_job_state != job_state:
                print(job)
                job_state = new_job_state
            else:
                time.sleep(10)

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

    # We have one process per GPU already
    # TODO: allow other devices
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    if partition == "debug":
        # Prevent double init when running in REPL
        # if not fairscale.nn.model_parallel.initialize.model_parallel_is_initialized():
        #     torch.distributed.init_process_group(backend="nccl")
        #     fairscale.nn.model_parallel.initialize_model_parallel(1)
        assert (
            num_gpus == 1
        ), "If you want more than one GPU, you need to specify a SLURM partition with --partition"
        log.info("Starting local training on 1 GPU.")
        return Env(workdir, 0, 1, torch.device("cuda:0"))

    # TODO: this assumes we are a slurm job, we might want to run on non-SLURM cluster
    env = submitit.helpers.TorchDistributedEnvironment()
    env.export()
    # TODO check how fairscale does it.
    os.environ["GROUP_RANK"] = "0"
    os.environ.update(os.environ)
    log.info(
        f"Starting distributed worker\nLOCAL_RANK: {env.local_rank}\n"
        f"RANK: {env.rank}\n"
        f"GROUP_RANK: {os.environ['GROUP_RANK']}\n"
        f"WORLD_SIZE: {env.world_size}"
    )

    torch.distributed.init_process_group(backend="nccl")
    # fairscale.nn.model_parallel.initialize_model_parallel(1)
    return Env(workdir, env.rank, env.world_size, device)


def parse_submitit_stderr(stderr: str, nlines: int = 10) -> str:
    stderr = stderr.rsplit("Submitted job triggered an exception\n", 1)[0]
    traces = stderr.split("Traceback (most recent call last):\n", 1)
    stderr, trace = traces if len(traces) == 2 else (stderr, "")
    last_stderr_lines = stderr.rsplit(os.linesep, nlines)
    last_stderr_lines.append(trace)
    return "\n".join(last_stderr_lines)
