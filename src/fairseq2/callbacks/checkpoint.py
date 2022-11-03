# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Set

import torch
from torchsnapshot import Snapshot
from torchtnt.runner.callback import Callback
from torchtnt.runner.callbacks import torchsnapshot_saver
from torchtnt.runner.state import State
from torchtnt.runner.unit import TrainUnit, TTrainData

log = logging.getLogger(__file__)


class WriteCheckpoint(Callback):
    """A callback to write prediction outputs to a CSV file.

    This callback provides an interface to simplify writing outputs during prediction
    into a CSV file. This callback must be extended with an implementation for
    ``get_batch_output_rows`` to write the desired outputs as rows in the CSV file.

    By default, outputs at each step across all processes will be written into the same CSV file.
    The outputs in each row is a a list of strings, and should match
    the columns names defined in ``header_row``.

    Args:
        header_row: columns of the CSV file
        dir_path: directory path of where to save the CSV file
        delimiter: separate columns in one row. Default is tab
        filename: name of the file. Default filename is "predictions.csv"
    """

    def __init__(
        self,
        folder: Path,
        frequency: timedelta,
        pattern: str = "checkpoint_{num_steps_completed}.pt",
    ) -> None:
        """
        :param frequency:
           Time between two checkpoints.
        """
        super().__init__()
        self.outdir = folder
        folder.mkdir(exist_ok=True)
        self.frequency = frequency
        self.last_save = datetime.now()
        self.pattern = pattern

    def on_train_step_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        step_end = datetime.now()
        elapsed = step_end - self.last_save
        if elapsed < self.frequency:
            return

        progress = state.train_state.progress  # type: ignore
        file = self.outdir / self.pattern.format(**progress.state_dict())
        log.info(f"Saving train state at {file} after {progress} steps and {elapsed}")
        torch.save(unit.state_dict(), file)
        self.last_save = datetime.now()
        log.info(f"Saving took {self.last_save - step_end}")
        checkpoint_last = self.outdir / "checkpoint_last.pt"
        if checkpoint_last.exists():
            checkpoint_last.unlink()
        checkpoint_last.symlink_to(file)


class TorchSnapshotLoader(Callback):
    def __init__(self, dirpath: str, *, replicated: List[str] = []):
        self.dirpath = dirpath
        self._replicated: Set[str] = set(replicated)

    def on_train_start(self, state: State, unit: TrainUnit[Any]) -> None:
        last_snapshot = resolve_last_snapshot(self.dirpath)
        if not last_snapshot:
            log.info("Starting new model training")
            return
        log.info(f"Resuming model training from {last_snapshot}")
        snapshot = Snapshot(path=str(last_snapshot))
        app_state = torchsnapshot_saver._get_app_state(
            state, unit, self._replicated, intra_epoch=True
        )
        snapshot.restore(app_state=app_state)


def resolve_last_snapshot(dirpath: str) -> Optional[Path]:
    snapshots = Path(dirpath).glob("epoch_*_step_*")
    try:
        return max(snapshots)
    except ValueError:
        # No snapshot found
        return None
