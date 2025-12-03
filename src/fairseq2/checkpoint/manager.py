# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from pickle import PickleError, PicklingError, UnpicklingError
from typing import Any, Protocol, final

import torch
from torch import Tensor
from torch.overrides import TorchFunctionMode
from typing_extensions import override

from fairseq2.device import CPU, CudaContext
from fairseq2.error import InternalError, StateDictError
from fairseq2.file_system import FileMode, FileSystem, _flush_nfs_lookup_cache
from fairseq2.gang import GangError, Gangs, all_sum
from fairseq2.io import (
    TensorFileDumper,
    TensorFileDumpOptions,
    TensorFileLoader,
    TensorFileLoadOptions,
)
from fairseq2.nn.fsdp import load_with_sdp_gang
from fairseq2.runtime.closable import Closable
from fairseq2.typing import Stateful
from fairseq2.utils.threading import ThreadPool


@dataclass
class CheckpointCallbackArgs:
    step_nr: int
    blocking: bool


class CheckpointCallback(Protocol):
    def __call__(self, args: CheckpointCallbackArgs) -> None: ...


@dataclass(kw_only=True)
class CheckpointSaveOptions:
    ready_callback: CheckpointCallback | None = None
    saved_callback: CheckpointCallback | None = None
    blocking: bool = False


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
        options: CheckpointSaveOptions | None = None,
    ) -> None:
        """
        :raises BadCheckpointError:
        :raises CheckpointError:
        """

    @abstractmethod
    def save_model_only(
        self,
        step_nr: int,
        model: Stateful,
        options: CheckpointSaveOptions | None = None,
    ) -> None:
        """
        :raises BadCheckpointError:
        :raises CheckpointError:
        """

    @abstractmethod
    def maybe_complete_save_operation(self, *, blocking: bool = False) -> bool | None:
        """
        :raises CheckpointError:
        """

    @property
    @abstractmethod
    def step_nr(self) -> int | None: ...

    @abstractmethod
    def save_score(self, step_nr: int, score: float) -> None:
        """
        :raises CheckpointError:
        """

    @abstractmethod
    def load_trainer_state(self, step_nr: int, trainer: Stateful) -> None:
        """
        :raises CheckpointNotFoundError:
        :raises BadCheckpointError:
        :raises CheckpointError:
        """

    @abstractmethod
    def load_model_state(self, step_nr: int, model: Stateful) -> None:
        """
        :raises CheckpointNotFoundError:
        :raises BadCheckpointError:
        :raises CheckpointError:
        """

    @abstractmethod
    def load_optimizer_state(self, step_nr: int, optimizer: Stateful) -> None:
        """
        :raises CheckpointNotFoundError:
        :raises BadCheckpointError:
        :raises CheckpointError:
        """

    @abstractmethod
    def load_data_reader_state(self, step_nr: int, data_reader: Stateful) -> None:
        """
        :raises CheckpointNotFoundError:
        :raises BadCheckpointError:
        :raises CheckpointError:
        """

    @abstractmethod
    def load_scores(self) -> list[tuple[float, int]]:
        """
        :raises BadCheckpointScoreError:
        :raises CheckpointError:
        """

    @abstractmethod
    def delete_checkpoint(self, step_nr: int, *, keep_model: bool = False) -> None:
        """
        :raises CheckpointError:
        """

    @abstractmethod
    def has_checkpoint(self, *, exclude_model_only: bool = False) -> bool:
        """
        :raises CheckpointError:
        """

    @abstractmethod
    def get_step_numbers(self, *, exclude_model_only: bool = False) -> list[int]:
        """
        :raises CheckpointError:
        """

    @abstractmethod
    def maybe_get_last_step_number(
        self, *, exclude_model_only: bool = False
    ) -> int | None:
        """
        :raises CheckpointError:
        """

    @abstractmethod
    def get_stale_step_numbers(
        self,
        keep_last_n: int | None,
        keep_best_n: int | None,
        keep_every_n_steps: int | None,
    ) -> list[int]:
        """
        :raises BadCheckpointScoreError:
        :raises CheckpointError:
        """


class CheckpointError(Exception):
    pass


class CheckpointNotFoundError(CheckpointError):
    def __init__(self, step_nr: int, kind: str) -> None:
        super().__init__(f"`{kind}` checkpoint of step {step_nr} is not found")

        self.step_nr = step_nr
        self.kind = kind


class BadCheckpointError(CheckpointError):
    def __init__(self, step_nr: int, kind: str, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr
        self.kind = kind


class BadCheckpointScoreError(CheckpointError):
    def __init__(self, step_nr: int) -> None:
        super().__init__(
            f"checkpoint score of step {step_nr} is not a valid floating-point number"
        )

        self.step_nr = step_nr


@final
class StandardCheckpointManager(CheckpointManager):
    """Saves and loads training checkpoints on a file system."""

    def __init__(
        self,
        output_dir: Path,
        gangs: Gangs,
        file_system: FileSystem,
        tensor_file_loader: TensorFileLoader,
        tensor_file_dumper: TensorFileDumper,
        thread_pool: ThreadPool,
        cuda_context: CudaContext,
    ) -> None:
        checkpoint_dir = output_dir.joinpath("checkpoints")

        self._checkpoint_dir = checkpoint_dir
        self._gangs = gangs
        self._file_system = file_system
        self._tensor_file_loader = tensor_file_loader
        self._tensor_file_dumper = tensor_file_dumper
        self._thread_pool = thread_pool
        self._cuda_context = cuda_context
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
        options: CheckpointSaveOptions | None = None,
    ) -> None:
        self.maybe_complete_save_operation(blocking=True)

        records: list[_CheckpointRecord] = []

        self._begin_checkpoint(step_nr)

        self._record_trainer_state(step_nr, trainer, records)

        self._record_model_state(step_nr, model, records)

        self._record_optimizer_state(step_nr, optimizer, records)

        self._record_data_reader_state(step_nr, data_reader, records)

        self._do_save_checkpoint(step_nr, records, options)

    @override
    def save_model_only(
        self,
        step_nr: int,
        model: Stateful,
        options: CheckpointSaveOptions | None = None,
    ) -> None:
        self.maybe_complete_save_operation(blocking=True)

        records: list[_CheckpointRecord] = []

        self._begin_checkpoint(step_nr)

        self._record_model_state(step_nr, model, records)

        self._do_save_checkpoint(step_nr, records, options)

    @override
    def maybe_complete_save_operation(self, *, blocking: bool = False) -> bool | None:
        step_nr = self._step_nr

        if step_nr is None:
            return None

        if self._save_op is None:
            raise InternalError(f"``step_nr` is {step_nr}, but `save_op` is `None`.")

        gangs = self._gangs

        if blocking:
            committer = self._save_op.result()

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise CheckpointError(
                    f"failed to sync ranks after saving checkpoint of step {step_nr}"
                ) from ex
        else:
            try:
                if self._save_op.done():
                    num_done = all_sum(gangs.root, 1)
                else:
                    num_done = all_sum(gangs.root, 0)
            except GangError as ex:
                raise CheckpointError(
                    f"failed to retrieve number of ranks completed saving checkpoint of step {step_nr}"
                ) from ex

            if num_done != gangs.root.size:
                return False

            committer = self._save_op.result()

        committer()

        self._save_op = None

        self._step_nr = None

        return True

    @property
    @override
    def step_nr(self) -> int | None:
        return self._step_nr

    def _begin_checkpoint(self, step_nr: int) -> None:
        self.delete_checkpoint(step_nr)

        gangs = self._gangs

        tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

        if gangs.root.rank == 0:
            self._create_directory(tmp_step_dir)

    def _record_trainer_state(
        self, step_nr: int, trainer: Stateful, records: list[_CheckpointRecord]
    ) -> None:
        gangs = self._gangs

        pathname = f"step_{step_nr}.tmp/trainer/rank_{gangs.root.rank:02d}.pt"

        file = self._checkpoint_dir.joinpath(pathname)

        if gangs.root.rank == 0:
            self._create_directory(file.parent)

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
            self._create_directory(file.parent)

        state_dict = model.state_dict()

        for k, v in state_dict.items():
            if not isinstance(v, Tensor):
                raise BadCheckpointError(
                    step_nr, "model", f"`model` checkpoint of step {step_nr} is expected to contain only objects of type `{Tensor}`, but the value of key '{k}' is of type `{type(v)}` instead"  # fmt: skip
                )

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
            self._create_directory(file.parent)

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
            self._create_directory(file.parent)

        state_dict = data_reader.state_dict()

        record = _CheckpointRecord("data_reader", file, state_dict)

        records.append(record)

    def _do_save_checkpoint(
        self,
        step_nr: int,
        records: list[_CheckpointRecord],
        options: CheckpointSaveOptions | None,
    ) -> None:
        if options is None:
            options = CheckpointSaveOptions()

        self._sync_nfs_cache()

        blocking = options.blocking

        if not blocking:
            memo: dict[Tensor, Tensor] = {}

            for record in records:
                self._copy_state_dict_to_host(record, step_nr, memo)

            del memo

        ready_callback = options.ready_callback
        saved_callback = options.saved_callback

        if ready_callback is not None:
            args = CheckpointCallbackArgs(step_nr, blocking)

            ready_callback(args)

        def save() -> Callable[[], None]:
            nonlocal records

            self._save_state_files(step_nr, records)

            del records

            def commit() -> None:
                self._commit_checkpoint(step_nr)

                if saved_callback is not None:
                    args = CheckpointCallbackArgs(step_nr, blocking)

                    saved_callback(args)

            return commit

        if blocking:
            committer = save()

            committer()
        else:
            self._save_op = self._thread_pool.queue(save)

    def _copy_state_dict_to_host(
        self, record: _CheckpointRecord, step_nr: int, memo: dict[Tensor, Tensor]
    ) -> None:
        d2h_mode = _CheckpointDeviceToHostMode(memo)

        try:
            with d2h_mode:
                record.state_dict = deepcopy(record.state_dict)
        except (ValueError, RuntimeError, TypeError, PickleError) as ex:
            raise BadCheckpointError(
                step_nr, record.kind, f"failed to deepcopy `{record.kind}` checkpoint of step {step_nr}"  # fmt: skip
            ) from ex

        if d2h_mode.has_cuda:
            self._cuda_context.synchronize()

    def _save_state_files(self, step_nr: int, records: list[_CheckpointRecord]) -> None:
        for record in records:
            # Do not use pickle protocol v5 for model state because
            # `torch.load(..., weights_only=True)` cannot unpickle objects
            # serialized with v5. See:
            #   https://github.com/pytorch/pytorch/issues/118092
            dump_options = TensorFileDumpOptions(
                pickle_protocol=2 if record.kind == "model" else 5
            )

            try:
                self._tensor_file_dumper.dump(
                    record.state_dict, record.file, dump_options
                )
            except (PicklingError, EOFError) as ex:
                raise BadCheckpointError(
                    step_nr, record.kind, f"failed to save `{record.kind}` checkpoint of step {step_nr} to file '{record.file}'"  # fmt: skip
                ) from ex
            except OSError as ex:
                raise CheckpointError(f"failed to write file '{record.file}'") from ex

    def _commit_checkpoint(self, step_nr: int) -> None:
        gangs = self._gangs

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise CheckpointError(
                f"failed to sync ranks before committing checkpoint of step {step_nr}"
            ) from ex

        if gangs.root.rank == 0:
            tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

            step_dir = tmp_step_dir.with_suffix("")

            try:
                self._file_system.move(tmp_step_dir, step_dir)
            except OSError as ex:
                raise CheckpointError(
                    f"failed to move directory '{tmp_step_dir}' to '{step_dir}'"
                ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise CheckpointError(
                f"failed to sync ranks after committing checkpoint of step {step_nr}"
            ) from ex

    def _sync_nfs_cache(self) -> None:
        if not self._file_system.is_local:
            return

        gangs = self._gangs

        if gangs.root.rank == 0:
            _flush_nfs_lookup_cache(self._checkpoint_dir)

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise CheckpointError(
                "failed to sync ranks while flushing NFS lookup cache"
            ) from ex

        if gangs.root.rank != 0:
            _flush_nfs_lookup_cache(self._checkpoint_dir)

    @override
    def save_score(self, step_nr: int, score: float) -> None:
        gangs = self._gangs

        if gangs.root.rank == 0:
            score_file = self._checkpoint_dir.joinpath(f"scores/step_{step_nr}.txt")

            self._create_directory(score_file.parent)

            try:
                fp = self._file_system.open_text(score_file, mode=FileMode.WRITE)
                with fp:
                    fp.write(f"{score}\n")
            except OSError as ex:
                raise CheckpointError(f"failed to write file '{score_file}'") from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise CheckpointError(
                f"failed to sync ranks after saving score of step {step_nr}"
            ) from ex

    @override
    def load_trainer_state(self, step_nr: int, trainer: Stateful) -> None:
        gangs = self._gangs

        pathname = f"trainer/rank_{gangs.root.rank:02d}.pt"

        kind = "trainer"

        state_dict = self._load_state_dict(step_nr, kind, pathname)

        try:
            trainer.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError, StateDictError) as ex:
            raise BadCheckpointError(
                step_nr, kind, f"failed to restore `{kind}` state from checkpoint of step {step_nr}"  # fmt: skip
            ) from ex

    @override
    def load_model_state(self, step_nr: int, model: Stateful) -> None:
        gangs = self._gangs

        pathname = f"model/pp_{gangs.pp.rank:02d}/tp_{gangs.tp.rank:02d}/sdp_{gangs.sdp.rank:02d}.pt"

        kind = "model"

        state_dict = self._load_state_dict(step_nr, kind, pathname)

        try:
            model.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError, StateDictError) as ex:
            raise BadCheckpointError(
                step_nr, kind, f"failed to restore `{kind}` state from checkpoint of step {step_nr}"  # fmt: skip
            ) from ex

    @override
    def load_optimizer_state(self, step_nr: int, optimizer: Stateful) -> None:
        gangs = self._gangs

        pathname = f"optimizer/pp_{gangs.pp.rank:02d}/tp_{gangs.tp.rank:02d}/sdp_{gangs.sdp.rank:02d}.pt"

        kind = "optimizer"

        state_dict = self._load_state_dict(step_nr, kind, pathname)

        try:
            optimizer.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError, StateDictError) as ex:
            raise BadCheckpointError(
                step_nr, kind, f"failed to restore `{kind}` state from checkpoint of step {step_nr}"  # fmt: skip
            ) from ex

    @override
    def load_data_reader_state(self, step_nr: int, data_reader: Stateful) -> None:
        gangs = self._gangs

        pathname = f"data_reader/dp_{gangs.dp.rank:02d}.pt"

        kind = "data_reader"

        state_dict = self._load_state_dict(step_nr, kind, pathname)

        try:
            data_reader.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError, StateDictError) as ex:
            raise BadCheckpointError(
                step_nr, kind, f"failed to restore `{kind}` state from checkpoint of step {step_nr}"  # fmt: skip
            ) from ex

    def _load_state_dict(
        self, step_nr: int, kind: str, pathname: str
    ) -> dict[str, object]:
        gangs = self._gangs

        load_options = TensorFileLoadOptions(map_location=CPU, restrict=kind == "model")

        with load_with_sdp_gang(gangs):  # Required for `ShardedTensor`.
            file = self._checkpoint_dir.joinpath(f"step_{step_nr}/{pathname}")

            try:
                return self._tensor_file_loader.load(file, load_options)
            except (UnpicklingError, EOFError) as ex:
                raise BadCheckpointError(
                    step_nr, kind, f"failed to load `{kind}` checkpoint of step {step_nr} from file '{file}'"  # fmt: skip
                ) from ex
            except FileNotFoundError:
                raise CheckpointNotFoundError(step_nr, kind) from None
            except OSError as ex:
                raise CheckpointError(f"failed to read file '{file}'") from ex

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
            fs = self._file_system

            step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

            # Delete the temporary checkpoint directory if exists.
            tmp_step_dir = step_dir.with_suffix(".tmp")

            try:
                fs.remove_directory(tmp_step_dir)
            except FileNotFoundError:
                pass
            except OSError as ex:
                raise CheckpointError(
                    f"failed to delete directory '{tmp_step_dir}'"
                ) from ex

            # Delete the checkpoint.
            try:
                dir_exists = fs.exists(step_dir)
            except OSError as ex:
                raise CheckpointError(
                    f"failed to access directory '{step_dir}'"
                ) from ex

            if dir_exists:
                if keep_model:

                    def paths() -> Iterator[Path]:
                        try:
                            yield from fs.glob(step_dir, pattern="*")
                        except OSError as ex:
                            raise CheckpointError(
                                f"failed to glob directory '{step_dir}'"
                            ) from ex

                    for path in paths():
                        if path.stem in ("model", "hg", "hook"):
                            continue

                        try:
                            is_dir = fs.is_dir(path)
                        except OSError as ex:
                            raise CheckpointError(
                                f"failed to access path '{path}'"
                            ) from ex

                        if is_dir:
                            try:
                                fs.remove_directory(path)
                            except OSError as ex:
                                raise CheckpointError(
                                    f"failed to delete directory '{path}'"
                                ) from ex
                        else:
                            try:
                                fs.remove(path)
                            except OSError as ex:
                                raise CheckpointError(
                                    f"failed to delete file '{path}'"
                                ) from ex
                else:
                    try:
                        fs.remove_directory(step_dir)
                    except OSError as ex:
                        raise CheckpointError(
                            f"failed to delete directory '{step_dir}'"
                        ) from ex

            if not keep_model:
                # Delete the score file.
                score_file = self._checkpoint_dir.joinpath(f"scores/step_{step_nr}.txt")

                try:
                    fs.remove(score_file)
                except FileNotFoundError:
                    pass
                except OSError as ex:
                    raise CheckpointError(
                        f"failed to delete file '{score_file}'"
                    ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise CheckpointError(
                f"failed to sync ranks after deleting checkpoint of step {step_nr}"
            ) from ex

    @override
    def has_checkpoint(self, *, exclude_model_only: bool = False) -> bool:
        step_nrs = self.get_step_numbers(exclude_model_only=exclude_model_only)

        return len(step_nrs) > 0

    @override
    def get_step_numbers(self, *, exclude_model_only: bool = False) -> list[int]:
        step_nrs = []

        fs = self._file_system

        def step_dirs() -> Iterator[Path]:
            try:
                yield from fs.glob(self._checkpoint_dir, "step_*")
            except OSError as ex:
                raise CheckpointError(
                    f"failed to glob directory '{self._checkpoint_dir}'"
                ) from ex

        for step_dir in step_dirs():
            try:
                dir_exists = fs.exists(step_dir)
            except OSError as ex:
                raise CheckpointError(
                    f"failed to access directory '{step_dir}'"
                ) from ex

            if not dir_exists:
                continue

            try:
                step_nr = int(step_dir.name[5:])
            except ValueError:
                continue

            if exclude_model_only:
                trainer_dir = step_dir.joinpath("trainer")

                try:
                    dir_exists = fs.exists(trainer_dir)
                except OSError as ex:
                    raise CheckpointError(
                        f"failed to access directory '{trainer_dir}'"
                    ) from ex

                if dir_exists:
                    step_nrs.append(step_nr)
            else:
                step_nrs.append(step_nr)

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
            raise CheckpointError(
                "failed to sync ranks after retrieving list of stale checkpoints"
            ) from ex

        return [s for s in step_nrs if s not in non_stale_step_nrs]

    def _do_load_scores(self, step_nrs: list[int]) -> list[tuple[float, int]]:
        scores_dir = self._checkpoint_dir.joinpath("scores")

        scores = []

        for step_nr in step_nrs:
            score_file = scores_dir.joinpath(f"step_{step_nr}.txt")

            try:
                with self._file_system.open_text(score_file) as fp:
                    line = fp.readline()
            except FileNotFoundError:
                continue
            except OSError as ex:
                raise CheckpointError(f"failed to read file '{score_file}'") from ex

            try:
                score = float(line)
            except ValueError:
                raise BadCheckpointScoreError(step_nr) from None

            scores.append((score, step_nr))

        scores.sort(reverse=True)

        return scores

    def _create_directory(self, path: Path) -> None:
        try:
            self._file_system.make_directory(path)
        except OSError as ex:
            raise CheckpointError(f"failed to create directory '{path}'") from ex

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
