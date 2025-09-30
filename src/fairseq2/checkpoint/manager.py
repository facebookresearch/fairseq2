# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from os import scandir
from pathlib import Path
from pickle import PickleError
from typing import Any, Protocol, final

import torch
from torch import Tensor
from torch.overrides import TorchFunctionMode
from typing_extensions import override

from fairseq2.device import CPU
from fairseq2.error import (
    InternalError,
    OperationalError,
    StateDictError,
    raise_operational_system_error,
)
from fairseq2.file_system import FileMode, FileSystem
from fairseq2.gang import GangError, Gangs, all_sum, raise_operational_gang_error
from fairseq2.io import (
    TensorDataNotValidError,
    TensorDumper,
    TensorFileError,
    TensorLoader,
)
from fairseq2.nn.fsdp import load_with_sdp_gang
from fairseq2.runtime.closable import Closable
from fairseq2.typing import Stateful
from fairseq2.utils.threading import ThreadPool


class CheckpointManager(Closable):
    """Saves and loads training checkpoints."""

    @abstractmethod
    def save_checkpoint(
        self,
        step_nr: int,
        trainer: Stateful,
        model: Stateful,
        optimizer: Stateful,
        data_reader: Stateful,
        *,
        ready_callback: CheckpointReadyCallback | None = None,
        saved_callback: CheckpointSavedCallback | None = None,
        blocking: bool = False,
    ) -> None: ...

    @abstractmethod
    def save_model_only(
        self,
        step_nr: int,
        model: Stateful,
        *,
        ready_callback: CheckpointReadyCallback | None = None,
        saved_callback: CheckpointSavedCallback | None = None,
        blocking: bool = False,
    ) -> None: ...

    @abstractmethod
    def maybe_complete_save_operation(
        self, *, blocking: bool = False
    ) -> bool | None: ...

    @property
    @abstractmethod
    def is_saving(self) -> bool: ...

    @abstractmethod
    def save_score(self, step_nr: int, score: float) -> None: ...

    @abstractmethod
    def load_trainer_state(self, step_nr: int, trainer: Stateful) -> None: ...

    @abstractmethod
    def load_model_state(self, step_nr: int, model: Stateful) -> None: ...

    @abstractmethod
    def load_optimizer_state(self, step_nr: int, optimizer: Stateful) -> None: ...

    @abstractmethod
    def load_data_reader_state(self, step_nr: int, data_reader: Stateful) -> None: ...

    @abstractmethod
    def load_scores(self) -> list[tuple[float, int]]: ...

    @abstractmethod
    def delete_checkpoint(self, step_nr: int, *, keep_model: bool = False) -> None: ...

    @abstractmethod
    def has_checkpoint(self, *, exclude_model_only: bool = False) -> bool: ...

    @abstractmethod
    def get_step_numbers(self, *, exclude_model_only: bool = False) -> list[int]: ...

    @abstractmethod
    def maybe_get_last_step_number(
        self, *, exclude_model_only: bool = False
    ) -> int | None: ...

    @abstractmethod
    def get_stale_step_numbers(
        self,
        keep_last_n: int | None,
        keep_best_n: int | None,
        keep_every_n_steps: int | None,
    ) -> list[int]: ...


class CheckpointReadyCallback(Protocol):
    def __call__(self, step_nr: int, blocking: bool) -> None: ...


class CheckpointSavedCallback(Protocol):
    def __call__(self, step_nr: int, blocking: bool) -> None: ...


class CheckpointStateNotValidError(Exception):
    def __init__(self, step_nr: int, kind: str, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr
        self.kind = kind


class CheckpointNotFoundError(Exception):
    def __init__(self, step_nr: int, kind: str) -> None:
        super().__init__(f"`{kind}` checkpoint of step {step_nr} is not found.")

        self.step_nr = step_nr
        self.kind = kind


class CheckpointError(Exception):
    def __init__(self, step_nr: int, kind: str, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr
        self.kind = kind


@final
class StandardCheckpointManager(CheckpointManager):
    """Saves and loads training checkpoints on a file system."""

    def __init__(
        self,
        output_dir: Path,
        gangs: Gangs,
        file_system: FileSystem,
        tensor_loader: TensorLoader,
        tensor_dumper: TensorDumper,
        thread_pool: ThreadPool,
    ) -> None:
        self._checkpoint_dir = output_dir.joinpath("checkpoints")
        self._gangs = gangs
        self._file_system = file_system
        self._tensor_loader = tensor_loader
        self._tensor_dumper = tensor_dumper
        self._thread_pool = thread_pool
        self._save_op: Future[Callable[[], None]] | None = None
        self._step_nr: int | None = None

    @override
    def save_checkpoint(
        self,
        step_nr: int,
        trainer: Stateful,
        model: Stateful,
        optimizer: Stateful,
        data_reader: Stateful,
        *,
        ready_callback: CheckpointReadyCallback | None = None,
        saved_callback: CheckpointSavedCallback | None = None,
        blocking: bool = False,
    ) -> None:
        self.maybe_complete_save_operation(blocking=True)

        self._begin_checkpoint(step_nr)

        records: list[_CheckpointRecord] = []

        self._record_trainer_state(step_nr, trainer, records)

        self._record_model_state(step_nr, model, records)

        self._record_optimizer_state(step_nr, optimizer, records)

        self._record_data_reader_state(step_nr, data_reader, records)

        self._do_save_checkpoint(
            step_nr, records, ready_callback, saved_callback, blocking
        )

    @override
    def save_model_only(
        self,
        step_nr: int,
        model: Stateful,
        *,
        ready_callback: CheckpointReadyCallback | None = None,
        saved_callback: CheckpointSavedCallback | None = None,
        blocking: bool = False,
    ) -> None:
        self.maybe_complete_save_operation(blocking=True)

        self._begin_checkpoint(step_nr)

        records: list[_CheckpointRecord] = []

        self._record_model_state(step_nr, model, records)

        self._do_save_checkpoint(
            step_nr, records, ready_callback, saved_callback, blocking
        )

    @override
    def maybe_complete_save_operation(self, *, blocking: bool = False) -> bool | None:
        if self._save_op is None:
            return None

        if self._step_nr is None:
            raise InternalError(
                "An asynchronous save operation is in progress, but `step_nr` is `None`."
            )

        gangs = self._gangs

        if blocking:
            committer = self._save_op.result()

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise_operational_gang_error(ex)
        else:
            try:
                if self._save_op.running():
                    num_completed = all_sum(gangs.root, 0)
                else:
                    num_completed = all_sum(gangs.root, 1)
            except GangError as ex:
                raise_operational_gang_error(ex)

            if num_completed != gangs.root.size:
                return False

            committer = self._save_op.result()

        self._save_op = None

        self._step_nr = None

        committer()

        return True

    @property
    @override
    def is_saving(self) -> bool:
        return self._save_op is not None

    def _begin_checkpoint(self, step_nr: int) -> None:
        self.delete_checkpoint(step_nr)

        gangs = self._gangs

        tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

        if gangs.root.rank == 0:
            try:
                self._file_system.make_directory(tmp_step_dir)
            except OSError as ex:
                raise_operational_system_error(ex)

    def _record_trainer_state(
        self, step_nr: int, trainer: Stateful, records: list[_CheckpointRecord]
    ) -> None:
        gangs = self._gangs

        pathname = f"step_{step_nr}.tmp/trainer/rank_{gangs.root.rank:02d}.pt"

        file = self._checkpoint_dir.joinpath(pathname)

        if gangs.root.rank == 0:
            try:
                self._file_system.make_directory(file.parent)
            except OSError as ex:
                raise_operational_system_error(ex)

        state_dict = trainer.state_dict()

        record = _CheckpointRecord("trainer", file, state_dict)

        records.append(record)

    def _record_model_state(
        self, step_nr: int, model: Stateful, records: list[_CheckpointRecord]
    ) -> None:
        gangs = self._gangs

        if gangs.rdp.rank != 0:
            return

        pathname = f"step_{step_nr}.tmp/model/pp_{gangs.pp.rank:02d}/tp_{gangs.tp.rank:02d}/sdp_{gangs.sdp.rank:02d}.pt"

        file = self._checkpoint_dir.joinpath(pathname)

        if gangs.sdp.rank == 0:
            try:
                self._file_system.make_directory(file.parent)
            except OSError as ex:
                raise_operational_system_error(ex)

        state_dict = model.state_dict()

        for k, v in state_dict.items():
            if not isinstance(v, Tensor):
                msg = f"`model` checkpoint of step {step_nr} must contain only objects of type `{Tensor}`, but the value of the {k} key is of type `{type(v)}`."

                raise CheckpointStateNotValidError(step_nr, "model", msg)

        record = _CheckpointRecord("model", file, state_dict)

        records.append(record)

    def _record_optimizer_state(
        self, step_nr: int, optimizer: Stateful, records: list[_CheckpointRecord]
    ) -> None:
        gangs = self._gangs

        if gangs.rdp.rank != 0:
            return

        pathname = f"step_{step_nr}.tmp/optimizer/pp_{gangs.pp.rank:02d}/tp_{gangs.tp.rank:02d}/sdp_{gangs.sdp.rank:02d}.pt"

        file = self._checkpoint_dir.joinpath(pathname)

        if gangs.sdp.rank == 0:
            try:
                self._file_system.make_directory(file.parent)
            except OSError as ex:
                raise_operational_system_error(ex)

        state_dict = optimizer.state_dict()

        record = _CheckpointRecord("optimizer", file, state_dict)

        records.append(record)

    def _record_data_reader_state(
        self, step_nr: int, data_reader: Stateful, records: list[_CheckpointRecord]
    ) -> None:
        gangs = self._gangs

        if gangs.tp.rank != 0:
            return

        pathname = f"step_{step_nr}.tmp/data_reader/dp_{gangs.dp.rank:02d}.pt"

        file = self._checkpoint_dir.joinpath(pathname)

        if gangs.dp.rank == 0:
            try:
                self._file_system.make_directory(file.parent)
            except OSError as ex:
                raise_operational_system_error(ex)

        state_dict = data_reader.state_dict()

        record = _CheckpointRecord("data_reader", file, state_dict)

        records.append(record)

    def _do_save_checkpoint(
        self,
        step_nr: int,
        records: list[_CheckpointRecord],
        ready_callback: CheckpointReadyCallback | None,
        saved_callback: CheckpointSavedCallback | None,
        blocking: bool,
    ) -> None:
        self._sync_nfs_cache()

        if not blocking:
            memo: dict[Tensor, Tensor] = {}

            for record in records:
                self._copy_state_dict_to_host(record, step_nr, memo)

            del memo

        if ready_callback is not None:
            ready_callback(step_nr, blocking)

        def save() -> Callable[[], None]:
            nonlocal records

            self._save_state_files(step_nr, records)

            del records

            def commit() -> None:
                self._commit_checkpoint(step_nr)

                if saved_callback is not None:
                    saved_callback(step_nr, blocking)

            return commit

        if blocking:
            committer = save()

            committer()
        else:
            self._step_nr = step_nr

            try:
                self._save_op = self._thread_pool.queue(save)
            except RuntimeError as ex:
                self._step_nr = None

                raise OperationalError("A thread pool queue operation failed.") from ex

    def _copy_state_dict_to_host(
        self, record: _CheckpointRecord, step_nr: int, memo: dict[Tensor, Tensor]
    ) -> None:
        d2h_mode = _CheckpointDeviceToHostMode(memo)

        try:
            with d2h_mode:
                record.state_dict = deepcopy(record.state_dict)
        except (ValueError, RuntimeError, TypeError, PickleError) as ex:
            msg = f"`{record.kind}` checkpoint must contain primitive and pickeable objects only."

            raise CheckpointStateNotValidError(step_nr, record.kind, msg) from ex

        if d2h_mode.has_cuda:
            torch.cuda.synchronize()

    def _save_state_files(self, step_nr: int, records: list[_CheckpointRecord]) -> None:
        for record in records:
            # We do not use pickle protocol v5 for model state because
            # `torch.load(..., weights_only=True)` cannot unpickle objects
            # serialized with v5. See:
            #   https://github.com/pytorch/pytorch/issues/118092
            protocol = 2 if record.kind == "model" else 5

            try:
                self._tensor_dumper.dump(
                    record.state_dict, record.file, pickle_protocol=protocol
                )
            except TensorDataNotValidError as ex:
                msg = f"`{record.kind}` checkpoint must contain primitive and pickeable objects only."

                raise CheckpointStateNotValidError(step_nr, record.kind, msg) from ex
            except OSError as ex:
                raise_operational_system_error(ex)

    def _commit_checkpoint(self, step_nr: int) -> None:
        gangs = self._gangs

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        if gangs.root.rank == 0:
            tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

            step_dir = tmp_step_dir.with_suffix("")

            try:
                self._file_system.move(tmp_step_dir, step_dir)
            except OSError as ex:
                raise_operational_system_error(ex)

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

    def _sync_nfs_cache(self) -> None:
        if not self._file_system.is_local:
            return

        gangs = self._gangs

        if gangs.root.rank == 0:
            self._flush_nfs_lookup_cache()

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        if gangs.root.rank != 0:
            self._flush_nfs_lookup_cache()

    def _flush_nfs_lookup_cache(self) -> None:
        path = self._checkpoint_dir

        # Use the `opendir`/`readdir`/`closedir` trick to drop all cached NFS
        # LOOKUP results.
        while path != path.parent:
            try:
                it = scandir(path)
            except FileNotFoundError:
                path = path.parent

                continue
            except OSError:
                break

            try:
                next(it)
            except StopIteration:
                pass
            finally:
                it.close()

            break

    @override
    def save_score(self, step_nr: int, score: float) -> None:
        gangs = self._gangs

        if gangs.root.rank == 0:
            scores_dir = self._checkpoint_dir.joinpath("scores")

            try:
                self._file_system.make_directory(scores_dir)
            except OSError as ex:
                raise_operational_system_error(ex)

            score_file = scores_dir.joinpath(f"step_{step_nr}.txt")

            try:
                fp = self._file_system.open_text(score_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise_operational_system_error(ex)

            try:
                fp.write(f"{score}\n")
            except OSError as ex:
                raise_operational_system_error(ex)
            finally:
                fp.close()

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

    @override
    def load_trainer_state(self, step_nr: int, trainer: Stateful) -> None:
        gangs = self._gangs

        pathname = f"trainer/rank_{gangs.root.rank:02d}.pt"

        kind = "trainer"

        state_dict = self._load_state_dict(step_nr, kind, pathname)

        try:
            trainer.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError, StateDictError) as ex:
            msg = f"`{kind}` state cannot be restored from checkpoint of step {step_nr}."  # fmt: skip

            raise CheckpointError(step_nr, kind, msg) from ex

    @override
    def load_model_state(self, step_nr: int, model: Stateful) -> None:
        gangs = self._gangs

        pathname = f"model/pp_{gangs.pp.rank:02d}/tp_{gangs.tp.rank:02d}/sdp_{gangs.sdp.rank:02d}.pt"

        kind = "model"

        state_dict = self._load_state_dict(step_nr, kind, pathname)

        try:
            model.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError, StateDictError) as ex:
            msg = f"`{kind}` state cannot be restored from checkpoint of step {step_nr}."  # fmt: skip

            raise CheckpointError(step_nr, kind, msg) from ex

    @override
    def load_optimizer_state(self, step_nr: int, optimizer: Stateful) -> None:
        gangs = self._gangs

        pathname = f"optimizer/pp_{gangs.pp.rank:02d}/tp_{gangs.tp.rank:02d}/sdp_{gangs.sdp.rank:02d}.pt"

        kind = "optimizer"

        state_dict = self._load_state_dict(step_nr, kind, pathname)

        try:
            optimizer.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError, StateDictError) as ex:
            msg = f"`{kind}` state cannot be restored from checkpoint of step {step_nr}."  # fmt: skip

            raise CheckpointError(step_nr, kind, msg) from ex

    @override
    def load_data_reader_state(self, step_nr: int, data_reader: Stateful) -> None:
        gangs = self._gangs

        pathname = f"data_reader/dp_{gangs.dp.rank:02d}.pt"

        kind = "data_reader"

        state_dict = self._load_state_dict(step_nr, kind, pathname)

        try:
            data_reader.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError, StateDictError) as ex:
            msg = f"`{kind}` state cannot be restored from checkpoint of step {step_nr}."  # fmt: skip

            raise CheckpointError(step_nr, kind, msg) from ex

    def _load_state_dict(
        self, step_nr: int, kind: str, pathname: str
    ) -> dict[str, object]:
        gangs = self._gangs

        with load_with_sdp_gang(gangs):  # Required for `ShardedTensor`.
            file = self._checkpoint_dir.joinpath(f"step_{step_nr}/{pathname}")

            try:
                return self._tensor_loader.load(
                    file, map_location=CPU, restrict=kind == "model"
                )
            except TensorFileError as ex:
                msg = f"{file} of step {step_nr} is not a valid checkpoint file."

                raise CheckpointError(step_nr, kind, msg) from ex
            except FileNotFoundError:
                raise CheckpointNotFoundError(step_nr, kind) from None
            except OSError as ex:
                raise_operational_system_error(ex)

    @override
    def load_scores(self) -> list[tuple[float, int]]:
        step_nrs = self.get_step_numbers()
        if not step_nrs:
            return []

        return self._do_load_scores(step_nrs)

    @override
    def delete_checkpoint(self, step_nr: int, *, keep_model: bool = False) -> None:
        gangs = self._gangs

        if gangs.root.rank == 0:
            step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

            # Delete the temporary checkpoint directory if exists.
            tmp_step_dir = step_dir.with_suffix(".tmp")

            try:
                self._file_system.remove_directory(tmp_step_dir)
            except FileNotFoundError:
                pass
            except OSError as ex:
                raise_operational_system_error(ex)

            # Delete the checkpoint.
            try:
                step_dir_exists = self._file_system.exists(step_dir)
            except OSError as ex:
                raise_operational_system_error(ex)

            if step_dir_exists:
                try:
                    if keep_model:
                        for path in self._file_system.glob(step_dir, pattern="*"):
                            if path.name == "model" or path.name == "hg":
                                continue

                            if self._file_system.is_dir(path):
                                self._file_system.remove_directory(path)
                            else:
                                self._file_system.remove(path)
                    else:
                        self._file_system.remove_directory(step_dir)
                except OSError as ex:
                    raise_operational_system_error(ex)

            if not keep_model:
                # Delete the score file.
                score_file = self._checkpoint_dir.joinpath(f"scores/step_{step_nr}.txt")

                try:
                    self._file_system.remove(score_file)
                except FileNotFoundError:
                    pass
                except OSError as ex:
                    raise_operational_system_error(ex)

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

    @override
    def has_checkpoint(self, *, exclude_model_only: bool = False) -> bool:
        step_nrs = self.get_step_numbers(exclude_model_only=exclude_model_only)

        return len(step_nrs) > 0

    @override
    def get_step_numbers(self, *, exclude_model_only: bool = False) -> list[int]:
        step_nrs = []

        try:
            for step_dir in self._file_system.glob(self._checkpoint_dir, "step_*"):
                if not self._file_system.is_dir(step_dir):
                    continue

                try:
                    step_nr = int(step_dir.name[5:])
                except ValueError:
                    continue

                if exclude_model_only:
                    trainer_dir = step_dir.joinpath("trainer")

                    if self._file_system.exists(trainer_dir):
                        step_nrs.append(step_nr)
                else:
                    step_nrs.append(step_nr)
        except OSError as ex:
            raise_operational_system_error(ex)

        step_nrs.sort()

        return step_nrs

    @override
    def maybe_get_last_step_number(
        self, *, exclude_model_only: bool = False
    ) -> int | None:
        step_nrs = self.get_step_numbers(exclude_model_only=exclude_model_only)
        if step_nrs:
            return step_nrs[-1]

        return None

    @override
    def get_stale_step_numbers(
        self,
        keep_last_n: int | None,
        keep_best_n: int | None,
        keep_every_n_steps: int | None,
    ) -> list[int]:
        if keep_last_n is None and keep_best_n is None and keep_every_n_steps is None:
            return []

        step_nrs = self.get_step_numbers()
        if not step_nrs:
            return []

        non_stale_step_nrs = {step_nrs[-1]}  # never delete the last checkpoint.

        if keep_last_n is not None:
            if keep_last_n <= 0:
                raise ValueError("`keep_last_n` must be greater than or equal to 1.")

            non_stale_step_nrs.update(step_nrs[-keep_last_n:])

        if keep_best_n is not None:
            if keep_best_n <= 0:
                raise ValueError("`keep_best_n` must be greater than or equal to 1.")

            scores = self._do_load_scores(step_nrs)

            non_stale_step_nrs.update(step_nr for _, step_nr in scores[:keep_best_n])

        if keep_every_n_steps is not None:
            if keep_every_n_steps <= 0:
                raise ValueError(
                    "`keep_every_n_steps` must be greater than or equal to 1."
                )

            non_stale_step_nrs.update(
                n for n in step_nrs if n % keep_every_n_steps == 0
            )

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        return [s for s in step_nrs if s not in non_stale_step_nrs]

    def _do_load_scores(self, step_nrs: list[int]) -> list[tuple[float, int]]:
        scores_dir = self._checkpoint_dir.joinpath("scores")

        scores = []

        for step_nr in step_nrs:
            score_file = scores_dir.joinpath(f"step_{step_nr}.txt")

            try:
                fp = self._file_system.open_text(score_file)
            except FileNotFoundError:
                continue
            except OSError as ex:
                raise_operational_system_error(ex)

            try:
                line = fp.readline()
            except OSError as ex:
                raise_operational_system_error(ex)
            finally:
                fp.close()

            try:
                score = float(line)
            except ValueError:
                msg = f"Score of step {step_nr} cannot be parsed as floating-point number."

                raise CheckpointError(step_nr, "score", msg) from None

            scores.append((score, step_nr))

        scores.sort(reverse=True)

        return scores

    @override
    def close(self) -> None:
        pass


@dataclass
class _CheckpointRecord:
    kind: str
    file: Path
    state_dict: dict[str, object]


class _CheckpointDeviceToHostMode(TorchFunctionMode):
    def __init__(self, memo: dict[Tensor, Tensor]) -> None:
        self.memo = memo
        self.has_cuda = False

    def __torch_function__(  # type: ignore[override]
        self, func: Any, types: Any, args: Any, kwargs: Any = None
    ) -> Any:
        if func is Tensor.__deepcopy__:
            source = args[0]

            is_cuda = source.device.type == "cuda"
            if is_cuda:
                self.has_cuda = True

            target = torch.empty_like(source, device=CPU)

            target.copy_(source, non_blocking=is_cuda)

            self.memo[source] = target

            return target

        if kwargs is None:
            kwargs = {}

        return func(*args, **kwargs)
