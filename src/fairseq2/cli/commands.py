"""
Main entry point for fairseq2
"""
import datetime
import hashlib
import itertools
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import submitit
import torch
import torchtnt.framework as tnt

import fairseq2.callbacks
import fairseq2.distributed
from fairseq2.cli import Xp, XpScript
from fairseq2.data import StringLike
from fairseq2.tasks import Seq2Seq

logging.basicConfig(level=logging.INFO)
# TODO: train/evaluate should also setup logging to a specific experiment file
log = logging.getLogger("fairseq2.cli")


def sha_key(overrides: Iterable[str]) -> str:
    # TODO: breaking change, move this to Xp, and include the script/config hash
    # TODO: could we use nice name like W&B instead of hexdigests ?
    return hashlib.sha256(";".join(overrides).encode("utf-8")).hexdigest()[:8]


def train(
    script: Path,
    workdir: Optional[Path] = None,
    partition: str = "debug",
    num_gpus: int = 1,
    eval_freq: int = -1,
    restart: bool = False,
    max_steps: Optional[int] = None,
    overrides: List[str] = [],
) -> Dict[str, Any]:
    """Launches a training script.

    script: the training script to launch. It needs at least:
        - a "task" function that returns a fairseq2 task (or a torchtnt train unit)
        - a "train_data" function that returns a dataloader for the task
        - a "valid_data" function if --eval_freq is set.

    workdir: we will create an Xp dir there and put it the script and model snapshots.

    eval_freq: enable evaluation on the valid_data
    partition: run on SLURM using the given partition
    num_gpus: number of GPU to use (requires --partition)
    restart: start the training from scratch, ignoring existing checkpoints
    """
    slurm_args, overrides = _extract_slurm_args(overrides)
    if workdir is None:
        workdir = script.parent.resolve()
        if "/fairseq2/examples/" in str(script.resolve()):
            raise Exception(
                "We don't want to generate models inside the fairseq2 git repo. Specify a valid workdir with 'workdir=...'"
            )
    else:
        # Make a copy of script to workdir
        workdir = workdir.resolve()
        workdir.mkdir(exist_ok=True)
        xp_sha = sha_key(overrides)
        if workdir.name != xp_sha:
            workdir = workdir / xp_sha
            workdir.mkdir(exist_ok=True)
        workdir_script = workdir / script.name
        if workdir_script != script:
            workdir_script.write_bytes(script.read_bytes())
        script = workdir_script

    env = fairseq2.distributed.init(
        workdir, partition, num_gpus, one_file=False, slurm_args=slurm_args
    )
    xp = Xp(script, script.with_suffix(".yaml"), overrides)
    if restart and xp.config_file.exists():
        xp.config_file.unlink()

    entry_point = "train"

    # TODO: allow script to be a yaml file
    module = XpScript.from_script(script, overrides=overrides)
    _setup_module(module, env, xp, entry_point)

    # Dataloader may start subprocess.
    # Do this before having loaded the model
    # TODO: merge train_data and valid_data
    train_data = module.call_fn("train_data", caller=entry_point)
    eval_data = (
        module.call_fn("valid_data", caller=entry_point) if eval_freq > 0 else []
    )
    task = module.call_fn("task", caller=entry_point)

    train_state = tnt.init_fit_state(
        train_data,
        eval_data,
        max_steps=max_steps,
        evaluate_every_n_steps=eval_freq if eval_freq > 0 else None,
        evaluate_every_n_epochs=None,
    )

    module.serialize(xp.config_file)
    # Try to resume from the same workdir.
    if not restart:
        fairseq2.callbacks.load_from_last_snapshot(str(workdir), train_state, task)

    callbacks = module.call_fn("callbacks", caller="train")

    try:
        tnt.fit(train_state, task, callbacks=callbacks)
    except torch.cuda.OutOfMemoryError:  # type: ignore
        # TODO: make this handler configurable.
        # Could there be a tnt "on_oom_error" callback ?
        log.error("Cuda Out Of Memory Error !")
        log.warning(torch.cuda.memory_summary())
        for line in _cuda_mem_profile():
            log.warning(line)
        raise

    try:
        # logger isn't required by tnt, so let's be a bit more task agnostic.
        logger: Any = module.call_fn("logger", caller="evaluate")
        return getattr(logger, "_last_metrics", {})
    except Exception:
        return {}


def _cuda_mem_profile() -> List[str]:
    import collections
    import gc

    shapes = []
    cpu = torch.device("cpu")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                if obj.device != cpu:
                    shapes.append(
                        (
                            obj.element_size() * obj.nelement(),
                            tuple(obj.shape),
                            f"{obj.device}:{obj.dtype}",
                        )
                    )
        except:
            pass
    shape_counts = list(collections.Counter(shapes).items())
    mem_usage = [
        (size * count / 1024 / 1024, count, shape, device)
        for (size, shape, device), count in shape_counts
    ]
    mem_usage.sort(reverse=True)
    # Note that the number of small tensors is already reported by memory_summary.
    # Having the shape allows to backtrack to the code creating the tensor.
    return [
        f"{mb:.1f}MB occupied by {count} {shape} tensors on {device}"
        for (mb, count, shape, device) in mem_usage
        if mb > 10
    ]


def grid(
    script: Path,
    workdir: Path,
    partition: str,
    num_gpus: int = 1,
    eval_freq: int = -1,
    overrides: List[str] = [],
) -> None:
    """
    Launch multiple training on SLURM, using a grid-search over the given parameters.

    Use key=value1,value2 to iterate over several values for the argument 'key'.

    - workdir: Where to create exp folder. Each experiment will have its own sub-folder
    - partition: SLURM partition to use
    - num_gpus: Number of gpus to use for each run
    - eval_freq: Evaluation frequency
    """
    workdir.mkdir(exist_ok=True)
    workdir = workdir.resolve()
    slurm_args, overrides = _extract_slurm_args(overrides)
    fixed_overrides = []
    grid_overrides = []
    for override in overrides:
        if "," in override:
            assert (
                "=" in override
            ), f"Can't parse override: {override}. Missing '='. Expected syntax is: key=value1,value2"
            key, raw_values = override.split("=", 1)
            grid_overrides.append(
                ["=".join((key, val)) for val in raw_values.split(",")]
            )
        else:
            fixed_overrides.append(override)

    experiments = itertools.product(*grid_overrides)

    # Those can be overriden by passing 'slurm.timeout=600' or 'slurm.cpus_per_task=8'
    cpus_per_gpu = 5
    num_gpu_per_node = 8
    default_timeout = 3 * 24 * 60
    ex = submitit.AutoExecutor(
        folder=workdir / "logs", cluster="local" if partition == "local" else "slurm"
    )
    # Launch job with one task per gpu
    ex.update_parameters(
        name=script.stem,
        nodes=max(num_gpus // num_gpu_per_node, 1),
        gpus_per_node=min(num_gpus, num_gpu_per_node),
        tasks_per_node=min(num_gpus, num_gpu_per_node),
        cpus_per_task=cpus_per_gpu,
        timeout_min=default_timeout,
        slurm_partition=partition,
        slurm_additional_parameters=slurm_args,
    )

    jobs = []
    with ex.batch():
        for xp in experiments:
            # TODO: we should validate experiments **BEFORE** sending them to the queue.
            # Otherwise we might wait a long time just to realize we made a typo.
            xp_overrides = fixed_overrides + list(xp)
            # Copy the script inside its future workdir.
            # The job can take some time to get scheduled, having a copy of the script
            # ensures that we aren't modifying before the job actually starts.
            xp_sha = sha_key(xp_overrides)
            xp_workdir = workdir / xp_sha
            xp_workdir.mkdir(exist_ok=True)
            xp_script = xp_workdir / script.name
            xp_script.write_bytes(script.read_bytes())

            job = ex.submit(
                train,
                xp_script,
                workdir=xp_workdir,
                eval_freq=eval_freq,
                overrides=xp_overrides,
            )
            jobs.append(job)

    print(f"Launched {len(jobs)} in job array on partition {partition}. {jobs[0]}")
    for job in submitit.helpers.as_completed(jobs):
        print(job)
        if job.exception() is not None:
            print(job.exception())


def evaluate(
    snapshot_dir: Path,
    partition: str = "debug",
    num_gpus: int = 1,
    script: Optional[Path] = None,
    overrides: List[str] = [],
) -> Dict[str, Any]:
    """
    Loads a model from a snapshot dir and runs the corresponding evaluation.

    - snapshot_dir: the folder containing the model weights and hubconf.py
    - script: overrides the "hubconf.py" found in the model snapshot. This can have unexpected results.
    """
    import torchsnapshot

    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot {snapshot_dir} not found.")
    script = script or snapshot_dir / "hubconf.py"
    if not script.exists():
        raise FileNotFoundError(f"{script} not found !")
    train_config = snapshot_dir / "hubconf.yaml"
    assert train_config.exists(), f"{train_config} not found !"

    slurm_args, overrides = _extract_slurm_args(overrides)
    xp_sha = "_" + sha_key(overrides) if overrides else ""
    # Create a different yaml file to store the eval config
    # This will mostly be the same than train config,
    # but it won't have trainer specific info, and might have some overrides
    xp = Xp(script, snapshot_dir / f"evaluate{xp_sha}.yaml", overrides)

    env = fairseq2.distributed.init(
        snapshot_dir, partition, num_gpus, slurm_args=slurm_args
    )

    module = XpScript.from_script(
        script,
        overrides=overrides,
        yaml_config=xp.config_file if xp.config_file.exists() else train_config,
    )
    _setup_module(module, env, xp, "evaluate")

    task = module.call_fn("task", caller="evaluate")
    eval_data = module.call_fn("valid_data", caller="evaluate")
    callbacks = module.call_fn("callbacks", caller="evaluate")
    module.serialize(xp.config_file)

    eval_state = tnt.init_eval_state(dataloader=eval_data)
    log.info(f"Evaluating on {eval_data} ...")

    snapshot = torchsnapshot.Snapshot(path=str(snapshot_dir))
    # Also restore state.train_state.progress, so we can log eval results at the proper step
    eval_state._train_state = tnt.PhaseState(dataloader=[])
    state_dict = task.state_dict_for_inference()
    state_dict["train_progress"] = eval_state._train_state.progress
    snapshot.restore(state_dict)

    try:
        # logger isn't stricly required, and CLI errors already have been caught at this point.
        logger: Any = module.call_fn("logger", caller="evaluate")
    except Exception:
        logger = None

    # TODO: this is very Wandb specific, it should be done somewhere else.
    if isinstance(logger, fairseq2.callbacks.WandbLogger):
        wandb_group = module["job.wandb_group"]
        logger.prepare()
        try:
            logger._wandb_run.use_artifact(f"{wandb_group}:latest")
        except Exception:
            # The artifact may not be "ready" yet (not sure what that mean)
            pass

    tnt.evaluate(eval_state, task, callbacks=callbacks)
    # The eval_state metrics have been reset at this point, so we need to fetch
    # the last logged value from the logger.
    return getattr(logger, "_last_metrics", {})


def eval_server(
    snapshot_root: Path,
    partition: str = "debug",
    num_gpus: int = 1,
    timeout: datetime.timedelta = datetime.timedelta(minutes=10),
    script: Optional[Path] = None,
    overrides: List[str] = [],
) -> None:
    """Run 'evaluate' on each new snapshot that appear under a given folder

    - snapshot_root: the root folder to monitor
    - partition: partition to use for the evaluate run. Run locally by default.
    - num_gpus: number of gpus for eval
    - script: overrides the "hubconf.py" found in the model snapshot. This can have unexpected results.
    """
    if not snapshot_root.exists():
        raise FileNotFoundError(f"Root folder {snapshot_root} doesn't exist !")
    if script and not script.exists():
        raise FileNotFoundError(f"--script {script} doesn't exist !")

    slurm_args, overrides = _extract_slurm_args(overrides)
    xp_sha = "_" + sha_key(overrides) if overrides else ""

    def _logfile(snapshot: Path) -> Path:
        # Write logs above the snapshot folder, allowing to delete the snapshot
        # without losing the evaluation results
        return snapshot.parent / f"{snapshot.name}.eval{xp_sha}.log"

    def _find_new_snapshots(snapshot_root: Path, treated: Set[Path]) -> Set[Path]:
        warned = False
        while True:
            if not snapshot_root.exists():
                raise FileNotFoundError(
                    f"Folder {snapshot_root} doesn't exists anymore."
                )
            try:
                snapshots = {
                    s
                    for s in snapshot_root.glob("**/epoch_*_step_*")
                    if s.is_dir() and not _logfile(s).exists()
                }
            except FileNotFoundError:
                # This can happen if someone deleted a folder we were traversing
                continue
            snapshots -= treated
            if snapshots:
                return snapshots

            if not warned:
                print(f"No new snapshot found under {snapshot_root}")
                warned = True
            time.sleep(10)

    treated: Set[Path] = set()
    failed: Set[Path] = set()
    timed_out: Set[Path] = set()
    while True:
        queue = list(_find_new_snapshots(snapshot_root, treated))
        # Shuffle the queue otherwise we always prioritize the same run.
        random.shuffle(queue)
        pending = len(queue)
        for snapshot in queue:
            if not snapshot.exists():
                pending -= 1
                continue

            logfile = _logfile(snapshot)
            print(f"Starting evaluation of {snapshot}, logs at {logfile}")
            # Run in a subprocess for better isolation
            eval_cmd = [
                sys.executable,
                "-m",
                "fairseq2.cli",
                "evaluate",
                snapshot,
                f"--partition={partition}",
                f"--num_gpus={num_gpus}",
            ]
            if script:
                eval_cmd += ["--script", script]
            eval_cmd += overrides

            with logfile.open("w", encoding="utf-8") as o:
                try:
                    # TODO allow to run several of those in parallel when using the cluster as the backend
                    eval_process = subprocess.run(eval_cmd, stdout=o, stderr=o, timeout=timeout.total_seconds())  # type: ignore
                    pending -= 1
                    if eval_process.returncode == 0:
                        status = "Evaluated"
                        # TODO: output the metrics in a structured format
                        # tag "best" snapshot
                        treated.add(snapshot)
                    else:
                        status = "Failed"
                        failed.add(snapshot)
                except (subprocess.TimeoutExpired, OSError):
                    pending -= 1
                    status = "Timedout"
                    timed_out.add(snapshot)

            print(
                f"{status} {snapshot} (pending: {pending}, evaluated: {len(treated)}, failed: {len(failed)}, timed_out: {len(timed_out)})"
            )


beam_search_kwargs = {
    "beam_size": 2,
    "max_len": 128,
    "unk_penalty": 1.0,
}


def inference(
    snapshot_dir: Path,
    src_lang: str = "",
    tgt_lang: str = "",
    partition: str = "debug",
    batch_size: int = 16,
    num_gpus: int = 1,
) -> None:
    import fairseq2.generate

    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot {snapshot_dir} not found.")

    # Currently inference always run locally. This could be an issue for large model
    # TODO: allow distributed inference (this won't work with stdin/stdout)
    assert partition == "debug", "TODO: local inference is supported"

    env = fairseq2.distributed.init(snapshot_dir, partition, num_gpus)
    # Note: it's important to use torch.hub.load here,
    # so we don't make too many assumption on how people store the model.
    task: Seq2Seq = torch.hub.load(
        snapshot_dir, "fairseq2_hub", snapshot_dir, source="local", device=env.device
    )

    task.module.eval()

    tty = False
    # hasattr to handle patched sys.stdin
    if hasattr(sys.stdin, "fileno"):
        tty = os.isatty(sys.stdin.fileno())
    if tty:
        batch_size = 1
    strategy = fairseq2.generate.BeamSearchStrategy(
        vocab_info=task.tokenizer.vocab_info,
        **beam_search_kwargs,  # type: ignore
    )

    def gen(batch: List[str]) -> Sequence[StringLike]:
        if not batch:
            return []
        return strategy.generate_str(
            task.module,  # type: ignore[arg-type]
            task.tokenizer,
            batch,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=env.device,
        )

    batch = []
    if tty:
        print("> ", end="", flush=True)
    for line in sys.stdin:
        batch.append(line.strip())
        if len(batch) < batch_size:
            continue
        for translation in gen(batch):
            print(translation)
        if tty:
            print("> ", end="", flush=True)
        batch.clear()

    for translation in gen(batch):
        print(translation)


def help(script: Path, overrides: List[str] = []) -> None:
    module = XpScript.from_script(script, overrides=overrides)
    # TODO: how to document env, xp and entrypoint ?
    # _setup_module(module, env, xp, entry_point)
    # TODO: nice error message when some entry points don't exist (incomplete script)
    print(
        module.help(
            "task",
            "train_data",
            "valid_data",
            "callbacks",
            hidden=["env", "xp", "entry_point"],
        )
    )


def main(script: Union[Path, str, None] = None) -> None:
    import func_argparse as fa

    parsers = {
        "train": fa.func_argparser(train),
        "evaluate": fa.func_argparser(evaluate),
        "inference": fa.func_argparser(inference),
        "grid": fa.func_argparser(grid),
        "eval_server": fa.func_argparser(eval_server),
        "help": fa.func_argparser(help),
    }
    # TODO: push this to func_argparse
    with_overrides = []
    for name, parser in parsers.items():
        # Promote the first argument to positional argument
        if len(parser._actions) < 2:
            continue

        # If main is called from a script `fairseq2.cli.commands.main(__file__)`
        # we remove the script CLI arg, otherwise we convert it to a positional CLI arg
        if script and parser._actions[1].dest == "script":
            parser._actions.remove(parser._actions[1])
            parser.set_defaults(script=Path(script))
        elif parser._actions[1].default is None:

            parser._actions[1].option_strings = ()

        # Handle overrides separately, I'm not sure why nargs="*" doesn't work as expected
        override_action = [
            a for a in parser._actions if "--overrides" in a.option_strings
        ]
        if len(override_action) == 1:
            parser._actions.remove(override_action[0])
            with_overrides.append(name)

    main_parser = fa.multi_argparser(description=__doc__, **parsers)

    known_args, overrides = main_parser.parse_known_args()
    parsed_args = vars(known_args)
    if not parsed_args:
        # Show help for multi argparser receiving no arguments.
        main_parser.print_help()
        main_parser.exit()
    command = parsed_args.pop("__command")

    if command.__name__ in with_overrides:
        parsed_args["overrides"] = overrides
        typo_in_command = any(o.startswith("-") for o in overrides)
    else:
        typo_in_command = len(overrides) > 0

    if typo_in_command:
        # Redo the parsing so that we have the normal error message for unk args
        main_parser.parse_args()

    command(**parsed_args)


def _extract_slurm_args(overrides: List[str]) -> Tuple[Dict[str, str], List[str]]:
    # TODO: this feels like a hack
    slurm_argslist = [o for o in overrides if o.startswith("slurm.")]
    overrides = [o for o in overrides if not o.startswith("slurm.")]

    slurm_args = {}
    for a in slurm_argslist:
        a = a[len("slurm_") :]
        k, v = a.split("=", 1)
        slurm_args[k] = v
    return slurm_args, overrides


def _setup_module(
    module: XpScript, env: fairseq2.distributed.Env, xp: Xp, entry_point: str
) -> None:
    module["env"] = env
    module["xp"] = xp
    module["entry_point"] = entry_point
