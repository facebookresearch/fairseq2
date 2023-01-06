"""
Main entry point for fairseq2
"""
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torchtnt.framework as tnt

import fairseq2.callbacks
import fairseq2.distributed
from fairseq2.cli import DynamicModule
from fairseq2.tasks import Seq2Seq

log = logging.getLogger("fairseq2.cli")


def train(
    script: Path,
    workdir: Optional[Path] = None,
    partition: str = "debug",
    num_gpus: int = 1,
    eval_freq: int = -1,
    wandb_project: str = "",
    overrides: List[str] = [],
) -> None:
    """Launches a training script.

    script: the training script to launch. It needs at least:
        - a "task" function that returns a fairseq2 task (or a torchtnt train unit)
        - a "train_data" function that returns a dataloader for the task
        - a "valid_data" function if --eval_freq is set.

    workdir: we will copy the script there
    """
    if workdir is None:
        workdir = script.parent
        if "/fairseq2/examples/" in str(script.resolve()):
            raise Exception(
                "We don't want to generate models inside the fairseq2 git repo. Specify a valid workdir with 'workdir=...'"
            )
    else:
        # Make a copy of script to workdir
        workdir.mkdir(exist_ok=True)
        workdir_script = workdir / script.name
        workdir_script.write_bytes(script.read_bytes())
        script = workdir_script

    env = fairseq2.distributed.init(workdir, partition, num_gpus, one_file=False)

    # TODO: should "train" be fetched from the script ? this way it could be overriden.
    # TODO: allow script to be a yaml file
    module = DynamicModule.from_script(script, overrides=overrides)
    module["env"] = env

    # TODO: provide a default 'task' impl
    task = module.call_fn("task", caller="train")
    # TODO: this is a bit expensive, do it in tests
    _ = pickle.loads(pickle.dumps(task))

    eval_data = module.call_fn("valid_data", caller="train") if eval_freq > 0 else []
    train_state = tnt.init_fit_state(
        module.call_fn("train_data", caller="train"),
        eval_data,
        evaluate_every_n_steps=eval_freq if eval_freq > 0 else None,
        evaluate_every_n_epochs=None,
    )

    # TODO: explicitly reload state here
    module.serialize(workdir, script)

    # TODO: allow the script to override this.
    callbacks = fairseq2.callbacks.default_callbacks(
        task, env, wandb_project=wandb_project, reload_model=True, script=script
    )
    tnt.fit(train_state, task, callbacks=callbacks)


def grid(
    script: Path,
    workdir: Optional[Path] = None,
    partition: str = "debug",
    num_gpus: int = 1,
    eval_freq: int = -1,
    wandb_project: str = "",
    overrides: List[str] = [],
) -> None:
    # TODO: Here we should parse overrides differently and allow several values per key
    ...


def evaluate(
    snapshot_dir: Path,
    partition: str = "debug",
    num_gpus: int = 1,
    wandb_project: str = "",
    overrides: List[str] = [],
) -> None:
    # TODO: investigate why this can fail with: RuntimeError: No such operator fbgemm::new_managed_tensor
    import torchsnapshot

    assert snapshot_dir.exists(), f"Snapshot {snapshot_dir} not found."
    script = snapshot_dir / "hubconf.py"
    assert script.exists(), f"{script} not found !"

    env = fairseq2.distributed.init(snapshot_dir, partition, num_gpus)

    module = DynamicModule.from_script(script, overrides=overrides)
    # TODO: avoid using the workdir to store intermediary file like the SPM.
    module["env"] = env._replace(workdir=module["", "env"].workdir)

    task = module.call_fn("task", caller="train")

    eval_data = module.call_fn("valid_data", caller="train")
    eval_state = tnt.init_eval_state(dataloader=eval_data)
    log.info(f"Evaluating on {eval_data} ...")

    snapshot = torchsnapshot.Snapshot(path=str(snapshot_dir))
    snapshot.restore(task.state_dict_for_inference())

    # TODO: How to reload the training logger so we can log the eval results in the same place ?
    # Note: maybe the logger should actually be part of the train unit,
    # and inference could decide not to reload it, while evaluate could reload it.
    config = module._raw_args
    if wandb_project:
        logger: fairseq2.callbacks.MetricLogger = fairseq2.callbacks.WandbLogger(
            wandb_project, config
        )
    else:
        logger = fairseq2.callbacks.StdoutLogger()

    tnt.evaluate(
        eval_state, task, callbacks=[fairseq2.callbacks.LogMetrics(task, logger)]
    )


beam_search_kwargs = {
    "beam_size": 2,
    "max_len": 128,
    "unk_penalty": 1.0,
}


def inference(
    snapshot_dir: Path,
    src_bos: str = "",
    tgt_bos: str = "",
    partition: str = "debug",
    batch_size: int = 16,
    num_gpus: int = 1,
) -> None:
    import fairseq2.generate

    assert snapshot_dir.exists(), f"Snapshot {snapshot_dir} not found."
    # Currently inference always run locally. This could be an issue for large model
    # TODO: allow distributed inference (this won't work with stdin/stdout)
    assert partition == "debug", "TODO: only single GPU inference is supported"

    env = fairseq2.distributed.init(snapshot_dir, partition, num_gpus)
    # Note: it's important to use torch.hub.load here,
    # so we don't make too many assumption on how people store the model.
    task: Seq2Seq = torch.hub.load(
        snapshot_dir, "hub_task", snapshot_dir, source="local", device=env.device
    )

    task.model.eval()

    tty = os.isatty(sys.stdin.fileno())
    if tty:
        batch_size = 1
    strategy = fairseq2.generate.BeamSearchStrategy(
        token_meta=task.tokenizer,
        **beam_search_kwargs,  # type: ignore
    )

    def gen(batch: List[str]) -> List[str]:
        if not batch:
            return batch
        return strategy.generate_str(
            task.model,
            task.tokenizer,
            batch,
            src_bos=src_bos,
            tgt_bos=tgt_bos,
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
