# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Set, Union

from torchsnapshot.snapshot import PendingSnapshot, Snapshot
from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks import torchsnapshot_saver
from torchtnt.framework.state import State
from torchtnt.framework.unit import EvalUnit, TrainUnit
from torchtnt.utils import rank_zero_info

log = logging.getLogger(__name__)


class TorchSnapshotSaver(Callback):
    """
    Periodically saves the application state during training using TorchSnapshot: https://pytorch.org/torchsnapshot.

    This callback supplements the application state provided by :class:`torchtnt.unit.AppStateMixin`
    with the train progress, train dataloader (if applicable), and random number generator state.

    It makes two different kind of snapshots:
    * training snapshots are considered temporary snapshot to protect against job crash.
    Will keep only a few of them.
    * eval snapshots happen at the end of each eval and we store them with some metrics.
    We keep more of them and the one with best performance will always be kept.

    Args:
        dirpath: Parent directory to save snapshots to.
        frequency: Time spent between different snapshots
        keep_train_snapshots: how many training snapshots to keep on disk
        keep_eval_snapshots: how many eval snapshots to keep on disk.
    """

    def __init__(
        self,
        savedir: Union[Path, str],
        *,
        frequency: timedelta = timedelta(minutes=20),
        keep_train_snapshots: int = 2,
        keep_eval_snapshots: int = 10,
    ) -> None:
        self.savedir = Path(savedir)
        self.frequency = frequency
        self.keep_train_snapshots = keep_train_snapshots
        self.keep_eval_snapshots = keep_eval_snapshots

        self._last_save = datetime.now()
        self._best_snapshot: Optional[Path] = None
        self._replicated: Set[str] = set()
        self._pending: Optional[PendingSnapshot] = None
        # List of snapshots pending cleanup.
        # TODO: persist this across restarts
        self._last_train_snapshots: List[Path] = []
        self._last_eval_snapshots: List[Path] = []
        self._ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def on_train_start(self, state: State, unit: TrainUnit[Any]) -> None:
        """Validate there's no key collision for the app state."""
        app_state = unit.app_state()
        torchsnapshot_saver._check_app_state_collision(app_state)
        try:
            self._replicated = set(unit.replicated_keys())
        except Exception:
            raise Exception(
                "To use fairseq2.callbacks.TorchSnapshotSaver you need to implement a 'replicated_keys()' method on your TrainUnit"
            )

    def save(
        self,
        snapshot_path: Path,
        state: State,
        unit: Any,
        *,
        force: bool,
    ) -> bool:
        if self._pending is not None:
            if self._pending.path == snapshot_path:
                # Snapshot for this step already has been saved.
                # This can happen when on_train_step and on_valid_step trigger at the same step.
                return True

            pending = not self._pending.done()
            if pending and force:
                self._pending.wait()
            elif pending:
                log.warning(
                    f"Still writing previous snapshot, will skip this one. Consider increasing 'frequency' (current {self.frequency})"
                )
                return False

        # intra_epoch=True because we always want to save the dataloader state.
        app_state = torchsnapshot_saver._get_app_state(
            state, unit, self._replicated, intra_epoch=True
        )
        self._pending = Snapshot.async_take(
            str(snapshot_path), app_state=app_state, replicated=list(self._replicated)
        )
        rank_zero_info(f"Saving snapshot to path: {snapshot_path}", logger=log)
        self._last_save = datetime.now()
        return True

    def on_train_step_end(self, state: State, unit: TrainUnit[Any]) -> None:
        step_end = datetime.now()
        elapsed = step_end - self._last_save
        if elapsed < self.frequency:
            return

        train_state = state.train_state
        assert train_state
        epoch = train_state.progress.num_epochs_completed
        step = train_state.progress.num_steps_completed
        snapshot_path = self.savedir / f"epoch_{epoch}_step_{step}.train"
        if self.save(snapshot_path, state, unit, force=False):
            self.rm_oldest(
                snapshot_path, self._last_train_snapshots, self.keep_train_snapshots
            )

    def on_eval_end(self, state: State, unit: EvalUnit[Any]) -> None:
        train_state = state.train_state
        assert train_state

        epoch = train_state.progress.num_epochs_completed
        step = train_state.progress.num_steps_completed
        snapshot_path = self.savedir / f"epoch_{epoch}_step_{step}"
        assert self.save(snapshot_path, state, unit, force=True)

        assert state.eval_state is not None
        try:
            # TODO: upgrade once torchtnt has builtin support for metrics
            best: bool = train_state.metrics["best"]  # type: ignore
        except Exception:
            return

        if not best:
            # Note: we aren't pushing the best snapshot, because we don't want to delete it.
            self.rm_oldest(
                snapshot_path, self._last_eval_snapshots, self.keep_eval_snapshots
            )
            return

        log.info("{snapshot_path} is the best checkpoint so far")
        eval_best = self.savedir / "eval_best"
        if eval_best.exists():
            assert (
                eval_best.is_symlink()
            ), f"{eval_best} already exists and isn't a symlink, can't save best checkpoint"
            # eval_best is no longer pointing to the last snapshot.
            # let's put it back in the list of snapshots pending deletion.
            self.rm_oldest(
                eval_best.resolve(), self._last_eval_snapshots, self.keep_eval_snapshots
            )
            eval_best.unlink()
        eval_best.symlink_to(snapshot_path)

    def on_train_end(self, state: State, unit: Any) -> None:
        self._wait()

    def on_exception(self, state: State, unit: Any, exc: BaseException) -> None:
        self._wait()

    def _wait(self) -> None:
        if self._pending is not None:
            self._pending.wait()
        self._ex.shutdown(wait=True)

    def rm_oldest(self, new_snapshot: Path, queue: List[Path], limit: int) -> None:
        if limit == 0:
            return
        queue.append(new_snapshot)
        if len(queue) <= limit:
            return

        # Because of best snapshots, the insertion order may be different from steps order
        queue.sort(key=get_step)
        old_snapshot = queue.pop(0)
        log.info(f"Deleting {old_snapshot}")
        self._ex.submit(subprocess.run, ["rm", "-r", str(old_snapshot)], check=True)  # type: ignore


def get_step(snapshot_path: Path) -> int:
    step_ = snapshot_path.name.split(".")[0]
    step_ = step_.split("_")[-1]
    return int(step_)


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
    try:
        last_step, last_snapshot = max(
            (get_step(p), p) for p in Path(dirpath).glob("epoch_*_step_*")
        )
        return last_snapshot
    except ValueError:
        # No snapshot found
        return None
