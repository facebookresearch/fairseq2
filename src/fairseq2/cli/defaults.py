import datetime
import functools
import os
import sys
import warnings
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
    save_frequency: datetime.timedelta = datetime.timedelta(minutes=20),
    save_async: bool = True,
    gc_frequency: int = 10,
    log_frequency: int = 1000,
) -> List[tnt.callback.Callback]:
    """Default fairseq2 callbacks.

    - save_frequency: duration between two snapshot of the training model
    - save_async: use asynchronous write to disk when saving the model
    - gc_frequency: synchronize GC runs across GPUs
    - log_frequency: frequence of metric logging (in steps)
    """
    from fairseq2.callbacks import Debugger, LogMetrics, TorchSnapshotSaver

    callbacks: List[tnt.callback.Callback] = []

    if log_frequency > 0:
        callbacks.append(LogMetrics(logger, frequency_steps=log_frequency))

    if entry_point == "train" and save_frequency.total_seconds() > 0:
        callbacks.append(
            TorchSnapshotSaver(
                xp.script.parent,
                script=xp.script,
                frequency=save_frequency,
                async_snapshot=save_async,
            )
        )

    if env.world_size > 1 and gc_frequency > 0:
        # Synchronize GC runs across all nodes
        callbacks.append(tnt.callbacks.GarbageCollector(step_interval=gc_frequency))

    if os.isatty(sys.stdout.fileno()):
        callbacks.append(Debugger())

    return callbacks


def logger(
    xp: "Xp",
    entry_point: str,
    env: "Env",
    wandb_project: str = "",
    text_only: bool = False,
) -> "MetricLogger":
    """Where to log metrics (default is tensorboard)

    - wandb_project: use W&B instead
    - text_only: just write as text
    """
    import fairseq2.callbacks

    assert xp.script and xp.script.exists()
    config_file = xp.script.with_suffix(".yaml")

    warnings.filterwarnings(
        "ignore", r"^TypedStorage is deprecated", UserWarning, "torchsnapshot"
    )
    if text_only:
        return fairseq2.callbacks.StdoutLogger(config_file)
    elif wandb_project:
        return fairseq2.callbacks.WandbLogger(
            config_file,
            project=wandb_project,
            job_type=entry_point,
            group_id=xp.sha_key,
        )
    else:
        return fairseq2.callbacks.TensorBoardLogger(config_file)


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
