import datetime
import functools
import os
import sys
from typing import TYPE_CHECKING, Iterable, List

import torch
import torchtnt.framework as tnt

import fairseq2.tasks
from fairseq2.optim.lr_scheduler import MyleLR

if TYPE_CHECKING:
    from fairseq2.callbacks import MetricLogger
    from fairseq2.cli import Env, Xp
    from fairseq2.data.text import Tokenizer

_imports = set(locals().keys())

task = fairseq2.tasks.Seq2Seq


def tokenizer() -> "Tokenizer":
    """!!! No tokenizer specified !!!"""
    raise NotImplementedError("tokenizer should be implemented in the user script")


def train_data() -> Iterable[None]:
    """!!! No training data specified !!!"""
    raise NotImplementedError("train_data should be implemented in the user script")


def valid_data() -> Iterable[None]:
    """No valid data specified."""
    return []


def module() -> torch.nn.Module:
    """!!! No trainable module specified !!!"""
    raise NotImplementedError("module should be implemented in the user script")


def callbacks(
    logger: "MetricLogger",
    entry_point: str,
    env: "Env",
    xp: "Xp",
    gc_frequency: int = 10,
    save_frequency: datetime.timedelta = datetime.timedelta(minutes=5),
) -> List[tnt.callback.Callback]:
    """Default fairseq2 callbacks.

    - gc_frequency: synchronize GC runs across GPUs
    - save_frequency: duration between two snapshot of the training model
    """
    from fairseq2.callbacks import Debugger, LogMetrics, TorchSnapshotSaver

    callbacks: List[tnt.callback.Callback] = [LogMetrics(logger)]

    if entry_point == "train":
        callbacks.append(
            TorchSnapshotSaver(
                xp.script.parent, script=xp.script, frequency=save_frequency
            )
        )

    if env.world_size > 1 and gc_frequency > 0:
        # Synchronize GC runs across all nodes
        callbacks.append(tnt.callbacks.GarbageCollector(step_interval=10))

    if os.isatty(sys.stdout.fileno()):
        callbacks.append(Debugger())

    return callbacks


def logger(
    xp: "Xp", entry_point: str, tensorboard: bool = False, wandb_project: str = ""
) -> "MetricLogger":
    """Default fairseq2 logger

    - tensorboard: use tensorboard
    - wandb_project: enable W&B
    """
    import fairseq2.callbacks

    assert xp.script and xp.script.exists()
    config_file = xp.script.with_suffix(".yaml")
    if tensorboard:
        return fairseq2.callbacks.TensorBoardLogger(config_file)
    elif wandb_project:
        return fairseq2.callbacks.WandbLogger(
            config_file,
            project=wandb_project,
            job_type=entry_point,
            group_id=xp.sha_key,
        )
    else:
        return fairseq2.callbacks.StdoutLogger(config_file)


def optimizer(
    module: torch.nn.Module,
    weight_decay: float = 1e-5,
    lr: float = 5e-4,
    eps: float = 1e-6,
) -> torch.optim.Optimizer:
    """Pytorch optimizer."""
    return torch.optim.Adam(
        module.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=eps,
        weight_decay=weight_decay,
    )


lr_scheduler = functools.partial(MyleLR, num_warmup_steps=4000)
"""Learning Rate scheduler, MyleLR by default"""


__all__ = [x for x in locals() if not x.startswith("_") and x not in _imports]
