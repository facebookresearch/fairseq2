import datetime
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import torchtnt.framework as tnt
import yaml
from torchtnt.framework.unit import TTrainUnit
from torchtnt.utils.loggers import MetricLogger

import fairseq2
import fairseq2.distributed

from .checkpoint import TorchSnapshotLoader, TorchSnapshotSaver
from .debugger import Debugger
from .loggers import LogMetrics, StdoutLogger, WandbCsvWriter, WandbLogger
from .metrics import WER, Bleu, EffectiveThroughput, Metrics


def default_callbacks(
    task: TTrainUnit[Any],
    env: fairseq2.distributed.Env,
    *,
    reload_model: bool,
    wandb_project: str = "",
    gc_frequency: int = 10,
    save_frequency: datetime.timedelta = datetime.timedelta(minutes=5),
    script: Optional[Path] = None,
) -> List[tnt.callback.Callback]:

    if wandb_project:
        # Is this the best place to read the config ?
        config = (
            {}
            if script is None
            else yaml.load(script.with_suffix(".yaml").read_text(), Loader=yaml.Loader)
        )
        config.pop("env")
        logger: MetricLogger = WandbLogger(wandb_project, config)
    else:
        logger = StdoutLogger()

    callbacks = [
        TorchSnapshotSaver(env.workdir, script=script),
        LogMetrics(task, logger),
    ]
    if env.world_size > 1 and gc_frequency > 0:
        # Synchronize GC runs across all nodes
        callbacks.append(tnt.callbacks.GarbageCollector(step_interval=10))

    if reload_model:
        callbacks.append(
            TorchSnapshotLoader(str(env.workdir), replicated=task.replicated_keys())
        )
    if os.isatty(sys.stdout.fileno()):
        callbacks.append(Debugger())

    return callbacks


__all__ = [
    "default_callbacks",
    "Bleu",
    "Debugger",
    "EffectiveThroughput",
    "LogMetrics",
    "Metrics",
    "MetricLogger",
    "StdoutLogger",
    "TorchSnapshotLoader",
    "TorchSnapshotSaver",
    "WandbCsvWriter",
    "WandbLogger",
    "WER",
]
