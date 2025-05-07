# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Set
from concurrent.futures import Future
from multiprocessing.pool import Pool
from os import scandir
from pathlib import Path
from pickle import PickleError
from shutil import Error
from signal import SIG_IGN, SIGINT, signal
from typing import ClassVar, Protocol, TypeAlias, cast, final, runtime_checkable

import torch.multiprocessing as mp
from torch import Tensor
from typing_extensions import override

from fairseq2.device import CPU
from fairseq2.error import InternalError
from fairseq2.file_system import FileMode, FileSystem
from fairseq2.gang import Gang, GangError, Gangs, all_sum
from fairseq2.nn.data_parallel import load_with_sdp_gang
from fairseq2.typing import Closable
from fairseq2.utils.io import (
    TensorDumper,
    TensorDumpError,
    TensorLoader,
    TensorLoadError,
)
from fairseq2.utils.threading import ThreadPool


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None: ...


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
        metadata: dict[str, object] | None = None,
        state_processor: CheckpointStateProcessor | None = None,
        callback: CheckpointCallback | None = None,
        blocking: bool = False,
    ) -> None: ...

    @abstractmethod
    def save_model_only(
        self,
        step_nr: int,
        model: Stateful,
        *,
        state_processor: CheckpointStateProcessor | None = None,
        callback: CheckpointCallback | None = None,
        blocking: bool = False,
    ) -> None: ...

    @abstractmethod
    def maybe_complete_async_checkpoint(
        self, *, blocking: bool = False
    ) -> bool | None: ...

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
    def load_metadata(self, step_nr: int) -> dict[str, object] | None: ...

    @abstractmethod
    def get_model_path(self, step_nr: int) -> Path: ...

    @abstractmethod
    def load_scores(self) -> list[tuple[float, int]]: ...

    @abstractmethod
    def delete_checkpoint(self, step_nr: int) -> None: ...

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


CheckpointState: TypeAlias = dict[str, tuple[Path, dict[str, object]]]


class CheckpointStateProcessor(Protocol):
    def __call__(self, step_nr: int, state: CheckpointState) -> None: ...


class CheckpointCallback(Protocol):
    def __call__(self, step_nr: int) -> None: ...


@final
class FileCheckpointManager(CheckpointManager):
    """Saves and loads training checkpoints on a file system."""

    _checkpoint_dir: Path
    _gangs: Gangs
    _file_system: FileSystem
    _saver: CheckpointSaver
    _tensor_loader: TensorLoader
    _thread_pool: ThreadPool
    _save_op: Future[Callable[[], None]] | None
    _step_nr: int | None

    def __init__(
        self,
        checkpoint_dir: Path,
        gangs: Gangs,
        file_system: FileSystem,
        saver: CheckpointSaver,
        tensor_loader: TensorLoader,
        thread_pool: ThreadPool,
    ) -> None:
        try:
            self._checkpoint_dir = file_system.resolve(checkpoint_dir)
        except OSError as ex:
            raise CheckpointError(
                f"'{checkpoint_dir}' cannot be accessed. See the nested exception for details."
            ) from ex

        self._gangs = gangs

        self._file_system = file_system

        self._saver = saver

        self._tensor_loader = tensor_loader

        self._thread_pool = thread_pool

        self._save_op = None

        self._step_nr = None

    @override
    def save_checkpoint(
        self,
        step_nr: int,
        trainer: Stateful,
        model: Stateful,
        optimizer: Stateful,
        data_reader: Stateful,
        *,
        metadata: dict[str, object] | None = None,
        state_processor: CheckpointStateProcessor | None = None,
        callback: CheckpointCallback | None = None,
        blocking: bool = False,
    ) -> None:
        self.maybe_complete_async_checkpoint(blocking=True)

        state: CheckpointState = {}

        self._begin_checkpoint(step_nr)

        self._collect_trainer_state(step_nr, trainer, state)

        self._collect_model_state(step_nr, model, state)

        self._collect_optimizer_state(step_nr, optimizer, state)

        self._collect_data_reader_state(step_nr, data_reader, state)

        self._collect_metadata(step_nr, metadata, state)

        self._do_save_checkpoint(step_nr, state, state_processor, callback, blocking)

    @override
    def save_model_only(
        self,
        step_nr: int,
        model: Stateful,
        *,
        state_processor: CheckpointStateProcessor | None = None,
        callback: CheckpointCallback | None = None,
        blocking: bool = False,
    ) -> None:
        self.maybe_complete_async_checkpoint(blocking=True)

        state: CheckpointState = {}

        self._begin_checkpoint(step_nr)

        self._collect_model_state(step_nr, model, state)

        self._do_save_checkpoint(step_nr, state, state_processor, callback, blocking)

    @override
    def maybe_complete_async_checkpoint(self, *, blocking: bool = False) -> bool | None:
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
                raise CheckpointSaveError(
                    self._step_nr, f"The collective barrier after the checkpoint save operation of step {self._step_nr} has failed. See the nested exception for details."  # fmt: skip
                ) from ex
        else:
            try:
                if self._save_op.running():
                    num_completed = all_sum(gangs.root, 0)
                else:
                    num_completed = all_sum(gangs.root, 1)
            except GangError as ex:
                raise CheckpointSaveError(
                    self._step_nr, f"The checkpoint completion status of step {self._step_nr} cannot be communicated across processes. See the nested exception for details."  # fmt: skip
                ) from ex

            if num_completed != gangs.root.size:
                return False

            committer = self._save_op.result()

        self._save_op = None

        self._step_nr = None

        committer()

        return True

    def is_saving(self) -> bool:
        return self._save_op is not None

    def _begin_checkpoint(self, step_nr: int) -> None:
        try:
            self.delete_checkpoint(step_nr)
        except CheckpointError as ex:
            raise CheckpointSaveError(
                step_nr, f"The previous checkpoint of step {step_nr} cannot be deleted. See the nested exception for details."  # fmt: skip
            ) from ex

        gangs = self._gangs

        tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

        if gangs.root.rank == 0:
            try:
                self._file_system.make_directory(tmp_step_dir)
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The temporary '{tmp_step_dir}' checkpoint directory of step {step_nr} cannot be created. See the nested exception for details."  # fmt: skip
                ) from ex

        self._checkpoint_step_nr = step_nr

    def _collect_trainer_state(
        self, step_nr: int, trainer: Stateful, state: CheckpointState
    ) -> None:
        gangs = self._gangs

        trainer_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp/trainer")

        if gangs.root.rank == 0:
            try:
                self._file_system.make_directory(trainer_dir)
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The '{trainer_dir}' trainer directory of step {step_nr} cannot be created. See the nested exception for details."  # fmt: skip
                ) from ex

        rank = self._rank_to_str(gangs.root)

        trainer_file = trainer_dir.joinpath(f"state_{rank}.pt")

        state_dict = trainer.state_dict()

        state["trainer"] = (trainer_file, state_dict)

    def _collect_model_state(
        self, step_nr: int, model: Stateful, state: CheckpointState
    ) -> None:
        gangs = self._gangs

        if gangs.rdp.rank != 0:
            return

        if gangs.tp.size == 1:
            pathname = f"step_{step_nr}.tmp/model.pt"
        else:
            pathname = f"step_{step_nr}.tmp/model.{gangs.tp.rank:02d}.pt"

        model_path = self._checkpoint_dir.joinpath(pathname)

        if gangs.sdp.size == 1:
            model_file = model_path
        else:
            if gangs.sdp.rank == 0:
                try:
                    self._file_system.make_directory(model_path)
                except OSError as ex:
                    raise CheckpointSaveError(
                        step_nr, f"The '{model_path} model directory of step {step_nr} cannot be created. See the nested exception for details."  # fmt: skip
                    ) from ex

            model_file = model_path.joinpath(f"shard.{gangs.sdp.rank:02d}.pt")

        state_dict = {"model": model.state_dict(), "fs2": True}

        state["model"] = (model_file, state_dict)

    def _collect_optimizer_state(
        self, step_nr: int, optimizer: Stateful, state: CheckpointState
    ) -> None:
        gangs = self._gangs

        if gangs.rdp.rank != 0:
            return

        if gangs.tp.size == 1:
            pathname = f"step_{step_nr}.tmp/optimizer.pt"
        else:
            pathname = f"step_{step_nr}.tmp/optimizer.{gangs.tp.rank:02d}.pt"

        optimizer_path = self._checkpoint_dir.joinpath(pathname)

        if gangs.sdp.size == 1:
            optimizer_file = optimizer_path
        else:
            if gangs.sdp.rank == 0:
                try:
                    self._file_system.make_directory(optimizer_path)
                except OSError as ex:
                    raise CheckpointSaveError(
                        step_nr, f"The '{optimizer_path} optimizer directory of step {step_nr} cannot be created. See the nested exception for details."  # fmt: skip
                    ) from ex

            optimizer_file = optimizer_path.joinpath(f"shard.{gangs.sdp.rank:02d}.pt")

        state_dict = optimizer.state_dict()

        state["optimizer"] = (optimizer_file, state_dict)

    def _collect_data_reader_state(
        self, step_nr: int, data_reader: Stateful, state: CheckpointState
    ) -> None:
        gangs = self._gangs

        if gangs.tp.rank != 0:
            return

        data_reader_dir = self._checkpoint_dir.joinpath(
            f"step_{step_nr}.tmp/data_reader"
        )

        if gangs.dp.rank == 0:
            try:
                self._file_system.make_directory(data_reader_dir)
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The '{data_reader_dir}' data reader directory of step {step_nr} cannot be created. See the nested exception for details."  # fmt: skip
                ) from ex

        rank = self._rank_to_str(gangs.dp)

        data_reader_file = data_reader_dir.joinpath(f"state_{rank}.pt")

        state_dict = data_reader.state_dict()

        state["data_reader"] = (data_reader_file, state_dict)

    def _collect_metadata(
        self, step_nr: int, metadata: dict[str, object] | None, state: CheckpointState
    ) -> None:
        gangs = self._gangs

        if gangs.root.rank == 0 and metadata is not None:
            metadata_file = self._checkpoint_dir.joinpath(
                f"step_{step_nr}.tmp/metadata.pt"
            )

            state["metadata"] = (metadata_file, metadata)

    def _do_save_checkpoint(
        self,
        step_nr: int,
        state: CheckpointState,
        state_processor: CheckpointStateProcessor | None,
        callback: CheckpointCallback | None,
        blocking: bool,
    ) -> None:
        try:
            self._sync_nfs_cache()
        except GangError as ex:
            raise CheckpointSaveError(
                step_nr, f"The collective barrier within the NFS cache drop operation of step {step_nr} has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        if not blocking:
            host_state = {}

            memo: dict[Tensor, Tensor] = {}

            for kind, (file, state_dict) in state.items():
                try:
                    state_dict = self._move_state_dict_to_host(state_dict, memo)
                except (RuntimeError, ValueError, TypeError) as ex:
                    raise CheckpointSaveError(
                        step_nr, f"The '{kind}' state of step {step_nr} cannot be transferred to the host memory. See the nested exception for details."  # fmt: skip
                    ) from ex

                host_state[kind] = (file, state_dict)

            state = host_state

            del memo

        if state_processor is not None:
            state_processor(step_nr, state)

        def save() -> Callable[[], None]:
            nonlocal state

            self._saver.save(step_nr, state)

            del state

            self._copy_cc(step_nr)

            def commit() -> None:
                self._commit_checkpoint(step_nr)

                if callback is not None:
                    callback(step_nr)

            return commit

        if blocking:
            committer = save()

            committer()
        else:
            self._step_nr = step_nr

            self._save_op = self._thread_pool.queue(save)

    @staticmethod
    def _move_state_dict_to_host(
        state_dict: Mapping[str, object], memo: dict[Tensor, Tensor]
    ) -> dict[str, object]:
        def move_to_host(item: object) -> object:
            if item is None:
                return None

            if isinstance(item, Tensor):
                cpu_tensor = memo.get(item)
                if cpu_tensor is None:
                    if item.device.type == "cpu":
                        cpu_tensor = item.detach().clone()
                    else:
                        cpu_tensor = item.detach().cpu()

                    memo[item] = cpu_tensor

                return cpu_tensor

            if isinstance(item, (bool, int, float, str, Path)):
                return item

            if isinstance(item, Mapping):
                return {move_to_host(k): move_to_host(v) for k, v in item.items()}

            if isinstance(item, list):
                return [move_to_host(e) for e in item]

            if isinstance(item, tuple):
                return tuple(move_to_host(e) for e in item)

            if isinstance(item, Set):
                return {move_to_host(e) for e in item}

            raise ValueError(
                f"`state_dict` must contain only items of types `bool`, `int`, float`, `str`, `Path`, and `Tensor`, but contains at least one item of type `{type(item)}` instead."
            )

        return cast(dict[str, object], move_to_host(state_dict))

    def _copy_cc(self, step_nr: int) -> None:
        gangs = self._gangs

        if gangs.root.rank == 0:
            tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

            cc_dir = self._checkpoint_dir.joinpath("cc")

            try:
                self._file_system.copy_directory(cc_dir, tmp_step_dir)
            except FileNotFoundError:
                pass
            except (OSError, Error) as ex:
                raise CheckpointSaveError(
                    step_nr, f"The carbon copy directory cannot be copied to the '{tmp_step_dir}' directory of step {step_nr}. See the nested exception for details."  # fmt: skip
                ) from ex

    def _commit_checkpoint(self, step_nr: int) -> None:
        gangs = self._gangs

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise CheckpointSaveError(
                step_nr, f"The collective barrier before the checkpoint commit operation of step {step_nr} has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        if gangs.root.rank == 0:
            tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

            step_dir = tmp_step_dir.with_suffix("")

            try:
                self._file_system.move(tmp_step_dir, step_dir)
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The temporary '{tmp_step_dir}' checkpoint directory of step {step_nr} cannot be committed. See the nested exception for details."  # fmt: skip
                ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise CheckpointSaveError(
                step_nr, f"The collective barrier after the checkpoint commit operation of step {step_nr} has failed. See the nested exception for details."  # fmt: skip
            ) from ex

    def _sync_nfs_cache(self) -> None:
        if not self._file_system.is_local:
            return

        gangs = self._gangs

        if gangs.root.rank == 0:
            self._flush_nfs_lookup_cache()

        gangs.root.barrier()

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
                raise CheckpointSaveError(
                    step_nr, f"The '{scores_dir}' checkpoint scores directory cannot be created. See the nested exception for details."  # fmt: skip
                ) from ex

            score_file = scores_dir.joinpath(f"step_{step_nr}.txt")

            try:
                fp = self._file_system.open_text(score_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The score of step {step_nr} cannot be saved to the '{score_file}' file. See the nested exception for details."  # fmt: skip
                ) from ex

            try:
                fp.write(f"{score}\n")
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The score of step {step_nr} cannot be saved to the '{score_file}' file. See the nested exception for details."  # fmt: skip
                ) from ex
            finally:
                fp.close()

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise CheckpointSaveError(
                step_nr, f"The collective barrier after the score save operation of step {step_nr} has failed. See the nested exception for details."  # fmt: skip
            ) from ex

    @override
    def load_trainer_state(self, step_nr: int, trainer: Stateful) -> None:
        gangs = self._gangs

        rank = self._rank_to_str(gangs.root)

        try:
            state_dict = self._load_state_dict(
                step_nr, pathname=f"trainer/state_{rank}.pt"
            )
        except TensorLoadError as ex:
            raise CheckpointLoadError(
                step_nr, f"The trainer state of step {step_nr} cannot be loaded from the '{ex.path}' file. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            trainer.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError) as ex:
            raise CheckpointLoadError(
                step_nr, f"The trainer state of step {step_nr} cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

    @override
    def load_model_state(self, step_nr: int, model: Stateful) -> None:
        gangs = self._gangs

        if gangs.tp.size == 1:
            pathname = "model.pt"
        else:
            pathname = f"model.{gangs.tp.rank:02d}.pt"

        if gangs.sdp.size > 1:
            pathname = f"{pathname}/shard.{gangs.sdp.rank:02d}.pt"

        try:
            checkpoint = self._load_state_dict(step_nr, pathname)
        except TensorLoadError as ex:
            raise CheckpointLoadError(
                step_nr, f"The model state of step {step_nr} cannot be loaded from the '{ex.path}' file. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            state_dict = checkpoint["model"]
        except KeyError:
            raise CheckpointLoadError(
                step_nr, f"The model state of step {step_nr} is not valid."  # fmt: skip
            ) from None

        if not isinstance(state_dict, Mapping):
            raise CheckpointLoadError(
                step_nr, f"The model state of step {step_nr} is not valid."  # fmt: skip
            )

        try:
            model.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError) as ex:
            raise CheckpointLoadError(
                step_nr, f"The model state of step {step_nr} cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

    @override
    def load_optimizer_state(self, step_nr: int, optimizer: Stateful) -> None:
        gangs = self._gangs

        if gangs.tp.size == 1:
            pathname = "optimizer.pt"
        else:
            pathname = f"optimizer.{gangs.tp.rank:02d}.pt"

        if gangs.sdp.size > 1:
            pathname = f"{pathname}/shard.{gangs.sdp.rank:02d}.pt"

        try:
            state_dict = self._load_state_dict(step_nr, pathname)
        except TensorLoadError as ex:
            raise CheckpointLoadError(
                step_nr, f"The optimizer state of step {step_nr} cannot be loaded from the '{ex.path}' file. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            optimizer.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError) as ex:
            raise CheckpointLoadError(
                step_nr, f"The optimizer state of step {step_nr} cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

    @override
    def load_data_reader_state(self, step_nr: int, data_reader: Stateful) -> None:
        gangs = self._gangs

        rank = self._rank_to_str(gangs.dp)

        try:
            state_dict = self._load_state_dict(
                step_nr, pathname=f"data_reader/state_{rank}.pt"
            )
        except TensorLoadError as ex:
            raise CheckpointLoadError(
                step_nr, f"The data reader state of step {step_nr} cannot be loaded from the '{ex.path}' file. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            data_reader.load_state_dict(state_dict)
        except (ValueError, TypeError, RuntimeError) as ex:
            raise CheckpointLoadError(
                step_nr, f"The data reader state of step {step_nr} cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

    @override
    def load_metadata(self, step_nr: int) -> dict[str, object] | None:
        try:
            return self._load_state_dict(step_nr, pathname="metadata.pt")
        except TensorLoadError as ex:
            raise CheckpointLoadError(
                step_nr, f"The metadata of step {step_nr} cannot be loaded from the '{ex.path}' file. See the nested exception for details."  # fmt: skip
            ) from ex
        except CheckpointNotFoundError:
            return None

    def _load_state_dict(self, step_nr: int, pathname: str) -> dict[str, object]:
        gangs = self._gangs

        with load_with_sdp_gang(gangs):  # Required for `ShardedTensor`.
            file = self._checkpoint_dir.joinpath(f"step_{step_nr}/{pathname}")

            try:
                return self._tensor_loader.load(file, map_location=CPU, restrict=False)
            except FileNotFoundError:
                raise CheckpointNotFoundError(step_nr) from None

    @staticmethod
    def _rank_to_str(gang: Gang) -> str:
        if gang.size < 10:
            return f"{gang.rank}"

        if gang.size < 100:
            return f"{gang.rank:02d}"
        else:
            return f"{gang.rank:03d}"

    @override
    def get_model_path(self, step_nr: int) -> Path:
        gangs = self._gangs

        if gangs.tp.size == 1:
            pathname = f"step_{step_nr}/model.pt"
        else:
            pathname = f"step_{step_nr}/model.{gangs.tp.rank:02d}.pt"

        model_path = self._checkpoint_dir.joinpath(pathname)

        try:
            model_path_exists = self._file_system.exists(model_path)
        except OSError as ex:
            raise CheckpointError(
                f"The '{model_path}' path cannot be accessed. See the nested exception for details."
            ) from ex

        if not model_path_exists:
            raise CheckpointNotFoundError(step_nr)

        return model_path

    @override
    def load_scores(self) -> list[tuple[float, int]]:
        step_nrs = self.get_step_numbers()
        if not step_nrs:
            return []

        return self._do_load_scores(step_nrs)

    @override
    def delete_checkpoint(self, step_nr: int) -> None:
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
                raise CheckpointDeleteError(
                    step_nr, f"The temporary '{tmp_step_dir}' checkpoint directory of step {step_nr} cannot be deleted. See the nested exception for details."  # fmt: skip
                ) from ex

            # Delete the checkpoint.
            try:
                step_dir_exists = self._file_system.exists(step_dir)
            except OSError as ex:
                raise CheckpointDeleteError(
                    step_nr, f"The '{step_dir}' checkpoint directory of step {step_nr} cannot be accessed. See the nested exception for details."  # fmt: skip
                ) from ex

            if step_dir_exists:
                try:
                    self._file_system.remove_directory(step_dir)
                except OSError as ex:
                    raise CheckpointDeleteError(
                        step_nr, f"The '{step_dir}' checkpoint directory of step {step_nr} cannot be deleted. See the nested exception for details."  # fmt: skip
                    ) from ex

            # Delete the score file.
            score_file = self._checkpoint_dir.joinpath(f"scores/step_{step_nr}.txt")

            try:
                self._file_system.remove(score_file)
            except FileNotFoundError:
                pass
            except OSError as ex:
                raise CheckpointDeleteError(
                    step_nr, f"The '{score_file}' score file of step {step_nr} cannot be deleted. See the nested exception for details."  # fmt: skip
                ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise CheckpointError(
                f"The collective barrier after the checkpoint delete operation of step {step_nr} has failed. See the nested exception for details."
            ) from ex

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

                    # Make sure that the directory does not only contain the
                    # model.
                    if self._file_system.exists(trainer_dir):
                        step_nrs.append(step_nr)
                else:
                    step_nrs.append(step_nr)
        except OSError as ex:
            raise CheckpointError(
                f"The '{self._checkpoint_dir}' checkpoint directory cannot be traversed. See the nested exception for details."
            ) from ex

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
                "The collective barrier before the checkpoint delete operation has failed. See the nested exception for details."
            ) from ex

        return [s for s in step_nrs if s not in non_stale_step_nrs]

    def _do_load_scores(self, step_nrs: list[int]) -> list[tuple[float, int]]:
        scores_dir = self._checkpoint_dir.joinpath("scores")

        scores = []

        for step_nr in step_nrs:
            score_file = scores_dir.joinpath(f"step_{step_nr}.txt")

            def load_error() -> CheckpointError:
                return CheckpointError(
                    f"The score of step {step_nr} cannot be loaded from the '{score_file}' file. See the nested exception for details."
                )

            try:
                fp = self._file_system.open_text(score_file)
            except FileNotFoundError:
                continue
            except OSError as ex:
                raise load_error() from ex

            try:
                line = fp.readline()
            except OSError as ex:
                raise load_error() from ex
            finally:
                fp.close()

            try:
                score = float(line)
            except ValueError:
                raise CheckpointError(
                    f"The score of step {step_nr} cannot be parsed as a floating-point number."
                ) from None

            scores.append((score, step_nr))

        scores.sort(reverse=True)

        return scores

    @override
    def close(self) -> None:
        self._saver.close()


class CheckpointSaver(Closable):
    @abstractmethod
    def save(self, step_nr: int, state: CheckpointState) -> None: ...


@final
class InProcCheckpointSaver(CheckpointSaver):
    _tensor_dumper: TensorDumper

    def __init__(self, tensor_dumper: TensorDumper) -> None:
        self._tensor_dumper = tensor_dumper

    @override
    def save(self, step_nr: int, state: CheckpointState) -> None:
        _save_state_files(self._tensor_dumper, step_nr, state)

    @override
    def close(self) -> None:
        pass


@final
class OutOfProcCheckpointSaver(CheckpointSaver):
    _pool: Pool

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    @staticmethod
    def create(tensor_dumper: TensorDumper) -> OutOfProcCheckpointSaver:
        ctx = mp.get_context("spawn")

        # Do not allow the pool process to handle SIGINT. It will be gracefully
        # closed when `close()` is called.
        sig = signal(SIGINT, SIG_IGN)

        try:
            pool = ctx.Pool(1, _PoolProcess.init, (tensor_dumper,))
        except (RuntimeError, ValueError, PickleError) as ex:
            raise CheckpointError(
                "The checkpoint process pool cannot be initialized. See the nested exception for details."  # fmt: skip
            ) from ex
        finally:
            signal(SIGINT, sig)

        return OutOfProcCheckpointSaver(pool)

    @override
    def save(self, step_nr: int, state: CheckpointState) -> None:
        try:
            self._pool.apply(_PoolProcess.save_state_files, (step_nr, state))
        except RuntimeError as ex:
            raise CheckpointError(
                "The checkpoint process pool has failed to dispatch the save operation. See the nested exception for details."  # fmt: skip
            ) from ex

    @override
    def close(self) -> None:
        self._pool.close()

        self._pool.join()


class _PoolProcess:
    _tensor_dumper: ClassVar[TensorDumper | None] = None

    @staticmethod
    def init(tensor_dumper: TensorDumper) -> None:
        _PoolProcess._tensor_dumper = tensor_dumper

    @staticmethod
    def save_state_files(step_nr: int, state: CheckpointState) -> None:
        if _PoolProcess._tensor_dumper is None:
            raise InternalError("`_tensor_dumper` is `None`.")

        _save_state_files(_PoolProcess._tensor_dumper, step_nr, state)


def _save_state_files(
    tensor_dumper: TensorDumper, step_nr: int, state: CheckpointState
) -> None:
    for kind, (file, state_dict) in state.items():
        try:
            tensor_dumper.dump(state_dict, file)
        except TensorDumpError as ex:
            raise CheckpointSaveError(
                step_nr, f"The '{kind}' state of step {step_nr} cannot be saved to the '{ex.path}' file. See the nested exception for details."  # fmt: skip
            ) from ex


class CheckpointNotFoundError(Exception):
    step_nr: int

    def __init__(self, step_nr: int) -> None:
        super().__init__(f"No checkpoint found for step {step_nr}.")

        self.step_nr = step_nr


class CheckpointError(Exception):
    pass


class CheckpointSaveError(CheckpointError):
    step_nr: int

    def __init__(self, step_nr: int, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr


class CheckpointLoadError(CheckpointError):
    step_nr: int

    def __init__(self, step_nr: int, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr


class CheckpointDeleteError(CheckpointError):
    step_nr: int

    def __init__(self, step_nr: int, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr
