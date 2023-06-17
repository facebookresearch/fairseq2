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

import fairseq2.cli.callbacks
from fairseq2 import DOC_MODE
from fairseq2.cli import Xp, XpScript
from fairseq2.cli.api import Env, Seq2Seq
from fairseq2.cli.distributed import distributed_init
from fairseq2.data import StringLike
from fairseq2.sequence_generator import BeamSearchStrategy
from fairseq2.typing import Device

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s (%(asctime)s) - %(message)s",
)
log = logging.getLogger("fairseq2.cli")


def train(
    script: Path,
    workdir: Optional[Path] = None,
    partition: str = "",
    num_gpus: int = 1,
    eval_freq: int = -1,
    max_steps: Optional[int] = None,
    overrides: List[str] = [],
) -> Dict[str, Any]:
    """Launches a training script.

    - script: the training script to launch. It needs at least:
        - a "task" function that returns a fairseq2 task (or a torchtnt train unit)
        - a "train_data" function that returns a dataloader for the task
        - a "valid_data" function if --eval_freq is set.

    - workdir: we will create an Xp dir there and put it the script and model snapshots.
    - eval_freq: enable evaluation on the valid_data
    - num_gpus: number of GPUs to use
    - partition: run on SLURM using the given partition (runs locally otherwise)
        When using SLURM all sbatch options can be set by prefixing them with "slurm.": ``slurm.time=4320``
    """
    slurm_args, overrides = _extract_slurm_args(overrides)
    xp = Xp(script, script.with_suffix(".yaml"), overrides)
    if workdir is None:
        workdir = script.parent.resolve()
        if "/fairseq2/examples/" in str(script.resolve()):
            raise Exception(
                "We don't want to generate models inside the fairseq2 git repo. Specify a valid workdir with 'workdir=...'"
            )
    else:
        # Make a copy of script to workdir
        workdir = workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        if workdir.name != xp.sha_key:
            workdir = workdir / xp.sha_key
            workdir.mkdir(exist_ok=True)
        workdir_script = workdir / script.name
        if workdir_script != script:
            workdir_script.write_bytes(script.read_bytes())
            # update the xp with the new script
            script = workdir_script
            xp = Xp(script, script.with_suffix(".yaml"), overrides)

    # Check this before creating a SLURM job.
    last_snapshot = fairseq2.cli.callbacks.resolve_last_snapshot(workdir)
    if last_snapshot:
        log.warning(f"Found previous experiment at {last_snapshot}, continuing it.")
    else:
        log.info(f"Starting new experiment at {script}")

    env = distributed_init(workdir, partition, num_gpus, slurm_args=slurm_args)
    entry_point = "train"

    module = XpScript.from_script(script, overrides=overrides)
    _setup_module(module, env, xp, entry_point)

    # Create callbacks first, so you can have user code running on process start.
    callbacks = module.call_fn("callbacks", caller=entry_point)

    # Dataloader may start subprocess.
    # Do this before having loaded the model
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
    if last_snapshot:
        fairseq2.cli.callbacks.load_snapshot(last_snapshot, train_state, task)

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
    cpu = Device("cpu")
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
            # TODO: this will read "script" several time, but should be cached instead.
            xp_sha = Xp(script, Path("TBD.yaml"), xp_overrides).sha_key
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
        else:
            print(job.result())


def evaluate(
    script: Path,
    snapshot: str = "",
    partition: str = "",
    num_gpus: int = 1,
    overrides: List[str] = [],
) -> Dict[str, Any]:
    """
    Loads a model from a snapshot dir and runs the corresponding evaluation.

    - script: the python script
    - script: overrides the "hubconf.py" found in the model snapshot. This can have unexpected results.
    """
    import torchsnapshot

    if not script.exists():
        raise FileNotFoundError(f"{script} not found !")
    if not snapshot:
        snapshot_dir = script.parent
    elif Path(snapshot).exists():
        snapshot_dir = Path(snapshot)
    elif (script.parent / snapshot).exists():
        snapshot_dir = script.parent / snapshot
    else:
        raise FileNotFoundError(f"Snapshot {snapshot} not found.")

    script = script or snapshot_dir / "hubconf.py"
    train_config = snapshot_dir / "hubconf.yaml"
    assert train_config.exists(), f"{train_config} not found !"

    slurm_args, overrides = _extract_slurm_args(overrides)
    eval_sha = "_" + _overrides_sha(overrides) if overrides else ""
    # Create a different yaml file to store the eval config
    # This will mostly be the same than train config,
    # but it won't have trainer specific info, and might have some overrides
    xp = Xp(script, snapshot_dir / f"evaluate{eval_sha}.yaml", overrides)

    env = distributed_init(snapshot_dir, partition, num_gpus, slurm_args=slurm_args)

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

    task_snapshot = torchsnapshot.Snapshot(path=str(snapshot_dir))
    # Also restore state.train_state.progress, so we can log eval results at the proper step
    eval_state._train_state = tnt.PhaseState(dataloader=[])
    state_dict = task.state_dict_for_inference()
    state_dict["train_progress"] = eval_state._train_state.progress
    task_snapshot.restore(state_dict)

    tnt.evaluate(eval_state, task, callbacks=callbacks)

    # Return the last logged metrics. "logger" isn't strictly required,
    # for training/eval so we don't force it to be exist or to have a _last_metrics
    try:
        logger: Any = module.call_fn("logger", caller="evaluate")
        return logger._last_metrics  # type: ignore
    except Exception:
        return {}


def eval_server(
    snapshot_root: Path,
    partition: str = "",
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
    eval_sha = "_" + _overrides_sha(overrides) if overrides else ""

    def _logfile(snapshot: Path) -> Path:
        # Write logs above the snapshot folder, allowing to delete the snapshot
        # without losing the evaluation results
        return snapshot.parent / f"{snapshot.name}.eval{eval_sha}.log"

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


def _overrides_sha(overrides: Sequence[str]) -> str:
    return hashlib.sha256(";".join(overrides).encode("utf-8")).hexdigest()[:8]


beam_search_kwargs = {
    "beam_size": 2,
    "max_len": 128,
    "unk_penalty": 1.0,
}


def inference(
    snapshot_dir: Path,
    src_lang: str = "",
    tgt_lang: str = "",
    partition: str = "",
    batch_size: int = 16,
    num_gpus: int = 1,
) -> None:
    """(**experimental**) Starts the model in interactive mode"""
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot {snapshot_dir} not found.")

    # Currently inference always run locally. This could be an issue for large model
    # TODO: allow distributed inference (this won't work with stdin/stdout)
    if partition:
        raise NotImplementedError("only local inference is supported for now")

    env = distributed_init(snapshot_dir, partition, num_gpus)
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
    strategy = BeamSearchStrategy(
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
    """Show available hyperparameters for the given script."""
    module = XpScript.from_script(script, overrides=overrides)
    # TODO: how to document env, xp and entrypoint ?
    # _setup_module(module, env, xp, entry_point)
    print(
        module.help(
            "task",
            "train_data",
            "valid_data",
            "callbacks",
            hidden=["env", "xp", "entry_point"],
        )
    )


def test(
    script: Path,
    fn: str,
    overrides: List[str] = [],
    num_gpus: int = 0,
    entry_point: str = "test",
    partition: str = "",
    num_examples: int = 10_000,
) -> None:
    """
    Runs a specific function in your training script.

    - fn: the name of the function to run.
      When testing "train_data" or "valid_data", we will also measure the throughput of the dataloader.
    """
    slurm_args, overrides = _extract_slurm_args(overrides)
    xp = Xp(script, Path("_test.yaml"), overrides)
    if xp.config_file.exists():
        xp.config_file.unlink()
    module = XpScript.from_script(script, overrides=overrides)

    # TODO: does it make sense to use SLURM here ?
    # at this point why not just launch the full training ?
    env = distributed_init(script.parent, partition, num_gpus, slurm_args=slurm_args)
    _setup_module(module, env, xp, entry_point)
    result = module.call_fn(fn, caller=entry_point)

    if fn in ("train_data", "valid_data"):
        return _test_dataloader(result, num_examples, env.device)
    else:
        print("Success!", f"{fn}({' '.join(overrides)}) =", result)


def _test_dataloader(
    dataloader: Iterable[Any], num_examples: int, device: Device
) -> None:
    start_time = time.perf_counter()
    res = torch.tensor(0, device=device)

    def _acc(example: Any) -> Any:
        # Compute a sum of all inputs.
        # This basically makes sure the GPU is doing a non-zero amount of work.
        if isinstance(example, list):
            return sum(_acc(x) for x in example)
        elif isinstance(example, torch.Tensor):
            return example.to(dtype=torch.float32).mean().to(device)
        elif isinstance(example, str):
            return len(example)
        elif isinstance(example, (int, float)):
            return example
        return 0.0

    n = 0
    dataloader = itertools.islice(dataloader, num_examples)
    try:
        import tqdm

        dataloader = tqdm.tqdm(dataloader, total=num_examples)
    except ImportError:
        pass

    for example in dataloader:
        res = res * 0.99 + 0.01 * _acc(example)
        n += 1

    res.cpu().item()
    duration = time.perf_counter() - start_time

    log.info(f"Treated {n} batches in {duration}s: {n / duration:.3f} batch per second")


def main(script: Union[Path, str, None] = None) -> None:
    import func_argparse as fa

    parsers = {
        "train": fa.func_argparser(train),
        "evaluate": fa.func_argparser(evaluate),
        "inference": fa.func_argparser(inference),
        "grid": fa.func_argparser(grid),
        "eval_server": fa.func_argparser(eval_server),
        "help": fa.func_argparser(help),
        "test": fa.func_argparser(test),
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


def _setup_module(module: XpScript, env: Env, xp: Xp, entry_point: str) -> None:
    module["env"] = env
    module["xp"] = xp
    module["entry_point"] = entry_point


if DOC_MODE:
    # Document the 3 builtin fixtures.
    # The example values are shown in the doc.

    env: Env = Env(16, 9, Device("cpu"))
    """The distributed environment we are currently running in.

    Typically used in dataloader to read only a shard of the data,
    or to put the model on the right device.

    - world_size: Total number of worker process working together
    - global_rank: Unique id of this worker. Workers are numbered from 0 to ``world_size - 1``
    - device: Cuda device this worker should use.
    """

    xp: Xp = Xp(
        Path("examples/train_mt.py"),
        Path("/checkpoint/bob/cool_exp/train_mt.yaml"),
        ["lr=0.013", "train_data.batch_size=128"],
    )
    """Metadata about the current experiment.

    Typically used to output files in the right place.

    - script: path to the experiment script
    - config_file: a yaml file representing all the hyper-parameters used
    - overrides: list of hyper-parameters set from the CLI
    - sha_key: hash of the experiment script and its hyper-parameters
    """

    entry_point: str = "train"
    """The name of the fairseq2 command that was used to start this program.

    In general experiment scripts should avoid having different behavior
    between training and evaluation, it's mostly used by the logger to save the
    train and eval metrics in different places.
    """
