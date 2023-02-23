import datetime
import os
import sys
from typing import TYPE_CHECKING, List

import torchtnt.framework as tnt

import fairseq2.distributed
import fairseq2.tasks

if TYPE_CHECKING:
    from fairseq2.callbacks import MetricLogger
    from fairseq2.cli.module_loader import XP


task = fairseq2.tasks.Seq2Seq


def callbacks(
    logger: "MetricLogger",
    entry_point: str,
    env: fairseq2.distributed.Env,
    xp: "XP",
    gc_frequency: int = 10,
    save_frequency: datetime.timedelta = datetime.timedelta(minutes=5),
) -> List[tnt.callback.Callback]:
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


def logger(xp: "XP", entry_point: str, wandb_project: str = "") -> "MetricLogger":
    import fairseq2.callbacks

    assert xp.script and xp.script.exists()
    config_file = xp.script.with_suffix(".yaml")
    if wandb_project:
        return fairseq2.callbacks.WandbLogger(
            config_file,
            project=wandb_project,
            job_type=entry_point,
            group_id="-".join(xp.script.parent.parts[-2:]),
        )
    else:
        return fairseq2.callbacks.StdoutLogger(config_file)


# TODO: try getting read of __all__
__all__ = [
    "task",
    "callbacks",
    "logger",
]
