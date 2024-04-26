# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from os import scandir
from pathlib import Path
from pickle import PickleError
from shutil import rmtree
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    final,
)

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.nn import Module
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from fairseq2.gang import Gang
from fairseq2.models.utils.checkpoint import load_checkpoint
from fairseq2.nn.utils.module import (
    infer_device,
    load_state_dict,
    reset_non_persistent_buffers,
    to_empty,
)
from fairseq2.typing import CPU, META, Device, override


class CheckpointManager(ABC):
    """Saves and loads training checkpoints."""

    @abstractmethod
    def save_checkpoint(
        self,
        step_nr: int,
        checkpoint: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Save the checkpoint of the specified training step.

        :param step_nr:
            The number of the training step.
        :param checkpoint:
            The checkpoint to save.
        :param metadata:
            The checkpoint metadata (e.g. training configuration) to save.
        """

    @abstractmethod
    def load_checkpoint(self, step_nr: int) -> Dict[str, Any]:
        """Load the checkpoint of the specified training step.

        :param step_nr:
            The number of the training step.
        """

    @abstractmethod
    def load_last_checkpoint(self) -> Tuple[int, Dict[str, Any]]:
        """Load the last checkpoint in the training.

        :returns:
            - The number of the training step.
            - The checkpoint.
        """

    @abstractmethod
    def load_metadata(self, step_nr: int) -> Optional[Dict[str, Any]]:
        """Load the checkpoint metadata of the specified training step.

        :param step_nr:
            The number of the training step.
        """

    @abstractmethod
    def delete_checkpoint(self, step_nr: int, *, missing_ok: bool = False) -> None:
        """Delete the checkpoint of the specified training step.

        :param step_nr:
            The number of the training step.
        :param missing_ok:
            If ``True``, does not raise error if the checkpoint does not exists.
        """

    @abstractmethod
    def keep_last_n_checkpoints(self, n: int) -> None:
        """Delete all but the last ``n`` checkpoints."""

    @abstractmethod
    def save_consolidated_fsdp_model(self, step_nr: int, model: Module) -> None:
        """Save ``model`` with a ``state_dict`` consolidated from all processes.

        :param step_nr:
            The number of the training step.
        :param model:
            The model to save.
        """

    # compat
    @abstractmethod
    def save_consolidated_model(self, step_nr: int, model: Module) -> None:
        ...

    @abstractmethod
    def load_model(
        self, step_nr: int, out: Module, *, device: Optional[Device] = None
    ) -> None:
        """Load the model of the specified training step.

        :param step_nr:
            The number of the training step.
        :param out:
            The model to load.
        :param device:
            The device on which to load ``out`` if it is on the meta device;
            ignored otherwise.
        """

    @abstractmethod
    def load_last_model(self, out: Module, *, device: Optional[Device] = None) -> int:
        """Load the last model in the training.

        :param out:
            The model to load.
        :param device:
            The device on which to load ``out`` if it is on the meta device;
            ignored otherwise.

        :returns:
            The number of the training step.
        """

    @abstractmethod
    def has_checkpoint(
        self, step_nr: Optional[int] = None, *, with_model: bool = False
    ) -> bool:
        """Return ``True`` if the manager holds a checkpoint.

        :param step_nr:
            The number of the training step. If ``None``, returns ``True`` if
            the manager holds at least one checkpoint.
        :param with_model:
            If ``True``, only considers training steps with a saved model.
        """

    @abstractmethod
    def get_step_numbers(self, *, with_model: bool = False) -> List[int]:
        """Return the numbers of the training steps that have a checkpoint.

        :param with_model:
            If ``True``, only considers training steps with a saved model.
        """

    @abstractmethod
    def get_last_step_number(self, *, with_model: bool = False) -> Optional[int]:
        """Return the number of the last training step that has a checkpoint.

        :param with_model:
            If ``True``, only considers training steps with a saved model.
        """


@final
class FileCheckpointManager(CheckpointManager):
    """Saves and loads training checkpoints on a file system."""

    _checkpoint_dir: Path
    _root_gang: Gang
    _dp_gang: Gang
    _shard_suffix: str
    _distributed_fs: bool
    _model_key: str
    _replicated_keys: Set[str]

    def __init__(
        self,
        checkpoint_dir: Path,
        gang: Gang,
        *,
        dp_gang: Optional[Gang] = None,
        tp_gang: Optional[Gang] = None,
        distributed_fs: bool = True,
        model_key: str = "model",
        replicated_keys: Optional[Sequence[str]] = None,
    ) -> None:
        """
        :param checkpoint_dir:
            The base directory under which to store the checkpoints.
        :param gang:
            The gang to coordinate the checkpoint operations.
        :param dp_gang:
            The gang used for data parallelism.
        :param tp_gang:
            The gang used for tensor parallelism. Must be specified if ``dp_gang``
            is not ``None``.
        :param distributed_fs:
            If ``True``, the underlying file system of ``checkpoint_dir`` is
            considered distributed (e.g. NFS).
        :param model_key:
            The key of the model in provided checkpoints.
        :param replicated_keys:
            The keys in provided checkpoints whose values are replicated across
            all processes in the gang.
        """
        self._root_gang = gang

        self._dp_gang = gang

        self._shard_suffix = ""

        if dp_gang is not None and tp_gang is not None:
            self._dp_rank = dp_gang

            if tp_gang.size > 1:
                self._shard_suffix = f".{tp_gang.rank}"
        elif dp_gang is not None or tp_gang is not None:
            raise ValueError("`dp_gang` and `tp_gang` must be both specified.")

        self._distributed_fs = distributed_fs

        if distributed_fs:
            self._checkpoint_dir = checkpoint_dir
        else:
            self._checkpoint_dir = checkpoint_dir.joinpath(
                f"rank_{self._dp_gang.rank}{self._shard_suffix}"
            )

        self._model_key = model_key

        if replicated_keys is None:
            self._replicated_keys = set()
        else:
            self._replicated_keys = set(replicated_keys)

    # compat
    @property
    def replicated_keys(self) -> Set[str]:
        return self._replicated_keys

    # compat
    @replicated_keys.setter
    def replicated_keys(self, value: Set[str]) -> None:
        self._replicated_keys = value

    @override
    def save_checkpoint(
        self,
        step_nr: int,
        checkpoint: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        self.delete_checkpoint(step_nr, missing_ok=True)

        def raise_error(cause: Exception) -> NoReturn:
            raise RuntimeError(
                f"The checkpoint of training step {step_nr} cannot be saved. See nested exception for details."
            ) from cause

        tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

        if self._root_gang.rank == 0 or not self._distributed_fs:
            try:
                tmp_step_dir.mkdir(parents=True)
            except OSError as ex:
                raise_error(ex)

        self._root_gang.barrier()

        # Do not modify the argument in-place. In case we fail, it should stay
        # intact.
        rank_part = checkpoint.copy()

        if self._model_replicated():
            if (state_dict := rank_part.pop(self._model_key, None)) is not None:
                if self._dp_gang.rank == 0 or not self._distributed_fs:
                    model_file = tmp_step_dir.joinpath(f"model{self._shard_suffix}.pt")

                    try:
                        torch.save({"model": state_dict}, model_file)
                    except (RuntimeError, OSError, PickleError) as ex:
                        raise_error(ex)

                self._root_gang.barrier()

        # For non-distributed file systems, we ignore the replicated keys and
        # force each process to save the full checkpoint.
        if self._replicated_keys and self._distributed_fs:
            if self._dp_gang.rank == 0:
                replicated_part = {}

                if "*" in self._replicated_keys:
                    replicated_part, rank_part = rank_part, replicated_part
                else:
                    for key in self._replicated_keys:
                        try:
                            replicated_part[key] = rank_part.pop(key)
                        except KeyError:
                            pass

                if replicated_part:
                    replicated_file = tmp_step_dir.joinpath(
                        f"replicated{self._shard_suffix}.pt"
                    )

                    try:
                        torch.save(replicated_part, replicated_file)
                    except (RuntimeError, OSError, PickleError) as ex:
                        raise_error(ex)
            else:
                if "*" in self._replicated_keys:
                    rank_part.clear()
                else:
                    for key in self._replicated_keys:
                        try:
                            del rank_part[key]
                        except KeyError:
                            pass

            self._root_gang.barrier()

            # Check if anything is left to save for the rank.
            skip_rank = not rank_part
        else:
            skip_rank = False

        if not skip_rank:
            rank_file = tmp_step_dir.joinpath(
                f"rank_{self._dp_gang.rank}{self._shard_suffix}.pt"
            )

            try:
                torch.save(rank_part, rank_file)
            except (RuntimeError, OSError, PickleError) as ex:
                raise_error(ex)

            self._root_gang.barrier()

        if metadata is not None:
            if self._dp_gang.rank == 0 or not self._distributed_fs:
                metadata_file = tmp_step_dir.joinpath(
                    f"metadata{self._shard_suffix}.pt"
                )

                try:
                    torch.save(metadata, metadata_file)
                except (RuntimeError, OSError, PickleError) as ex:
                    raise_error(ex)

            self._root_gang.barrier()

        if self._root_gang.rank == 0 or not self._distributed_fs:
            try:
                tmp_step_dir.replace(tmp_step_dir.with_suffix(""))
            except OSError as ex:
                raise_error(ex)

        self._root_gang.barrier()

    @override
    def load_checkpoint(self, step_nr: int) -> Dict[str, Any]:
        def raise_error(cause: Exception) -> NoReturn:
            raise RuntimeError(
                f"The checkpoint of training step {step_nr} cannot be loaded. See nested exception for details."
            ) from cause

        parts = []

        filenames = [
            f"replicated{self._shard_suffix}.pt",
            f"rank_{self._dp_gang.rank}{self._shard_suffix}.pt",
        ]

        if self._model_replicated():
            filenames.append(f"model{self._shard_suffix}.pt")

        step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

        for filename in filenames:
            try:
                part = load_checkpoint(
                    step_dir.joinpath(filename), map_location=CPU, mmap=True
                )
            except FileNotFoundError:
                part = None
            except (RuntimeError, OSError, PickleError) as ex:
                raise_error(ex)

            if part is not None:
                # Restore the actual model key.
                if filename.startswith("model") and self._model_key != "model":
                    try:
                        part = {self._model_key: part["model"]}
                    except KeyError as ex:
                        raise_error(ex)

                parts.append(part)

            self._root_gang.barrier()

        if not parts:
            raise CheckpointNotFoundError(f"Training step {step_nr} has no checkpoint.")

        checkpoint = parts[0]

        # Merge the checkpoint parts together.
        for part in parts[1:]:
            checkpoint.update(part)

        return checkpoint

    def _model_replicated(self) -> bool:
        if self._dp_gang.size == 1:
            return True

        return self._model_key in self._replicated_keys or "*" in self._replicated_keys

    @override
    def load_last_checkpoint(self) -> Tuple[int, Dict[str, Any]]:
        last_step_nr = self.get_last_step_number()
        if last_step_nr is None:
            raise CheckpointNotFoundError("No checkpoint found.")

        # If we don't have a distributed file system, we have to ensure that we
        # have a consistent view of checkpoints across all processes.
        if not self._distributed_fs:
            gang = self._root_gang

            step_numbers = torch.empty(
                (gang.size,), device=gang.device, dtype=torch.int64
            )

            self._root_gang.all_gather(
                step_numbers, torch.tensor(last_step_nr, device=gang.device)
            )

            if not (step_numbers == last_step_nr).all():
                s = ", ".join(str(i) for i in step_numbers.tolist())

                raise RuntimeError(
                    f"The processes in the gang have no consensus on the last training step. The last step numbers sorted by rank: {s}"
                )

        checkpoint = self.load_checkpoint(last_step_nr)

        return last_step_nr, checkpoint

    @override
    def load_metadata(self, step_nr: int) -> Optional[Dict[str, Any]]:
        metadata_file = self._checkpoint_dir.joinpath(
            f"step_{step_nr}/metadata{self._shard_suffix}.pt"
        )

        try:
            metadata = load_checkpoint(metadata_file, map_location=CPU, mmap=True)
        except FileNotFoundError:
            metadata = None
        except (RuntimeError, OSError, PickleError) as ex:
            raise RuntimeError(
                f"The checkpoint metadata of training step {step_nr} cannot be loaded. See nested exception for details."
            ) from ex

        self._root_gang.barrier()

        return metadata

    @override
    def delete_checkpoint(self, step_nr: int, *, missing_ok: bool = False) -> None:
        if self._root_gang.rank == 0 or not self._distributed_fs:
            step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

            try:
                rmtree(step_dir)
            except OSError as ex:
                if not missing_ok or not isinstance(ex, FileNotFoundError):
                    raise RuntimeError(
                        f"The checkpoint of training step {step_nr} cannot be deleted. See nested exception for details."
                    ) from ex

            try:
                rmtree(step_dir.with_suffix(".tmp"))
            except OSError as ex:
                if not isinstance(ex, FileNotFoundError):
                    raise RuntimeError(
                        f"The checkpoint of training step {step_nr} cannot be deleted. See nested exception for details."
                    ) from ex

        self._root_gang.barrier()

    @override
    def keep_last_n_checkpoints(self, n: int) -> None:
        step_numbers = self.get_step_numbers()

        for step_number in step_numbers[:-n]:
            self.delete_checkpoint(step_number)

    @override
    def save_consolidated_fsdp_model(self, step_nr: int, model: Module) -> None:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()

        self._root_gang.barrier()

        if self._dp_gang.rank == 0:
            tmp_model_file = self._checkpoint_dir.joinpath(
                f"step_{step_nr}/model{self._shard_suffix}.tmp"
            )

            try:
                torch.save({"model": state_dict}, tmp_model_file)
            except (RuntimeError, OSError, PickleError) as ex:
                raise RuntimeError(
                    f"The model of training step {step_nr} cannot be saved. See nested exception for details."
                ) from ex

            try:
                tmp_model_file.replace(tmp_model_file.with_suffix(".pt"))
            except OSError as ex:
                raise RuntimeError(
                    f"The model of training step {step_nr} cannot be saved. See nested exception for details."
                ) from ex

        self._root_gang.barrier()

    # compat
    @override
    def save_consolidated_model(self, step_nr: int, model: Module) -> None:
        self.save_consolidated_fsdp_model(step_nr, model)

    @override
    def load_model(
        self, step_nr: int, out: Module, *, device: Optional[Device] = None
    ) -> None:
        model_file = self._checkpoint_dir.joinpath(
            f"step_{step_nr}/model{self._shard_suffix}.pt"
        )

        def raise_error(cause: Exception) -> NoReturn:
            raise RuntimeError(
                f"The model of training step {step_nr} cannot be loaded. See nested exception for details."
            ) from cause

        try:
            checkpoint = load_checkpoint(model_file, map_location=CPU, restrict=True)
        except FileNotFoundError:
            raise CheckpointNotFoundError(
                f"Training step {step_nr} has no saved model."
            )
        except (RuntimeError, OSError, PickleError) as ex:
            raise_error(ex)

        model_device = infer_device(out, name="out")

        if model_device == META:
            # Move the model to the actual device without initializing. Its
            # state will be overwritten by the checkpoint anyways.
            to_empty(out, device=device or CPU)

        # Load the model.
        try:
            state_dict = checkpoint["model"]
        except KeyError as ex:
            raise_error(ex)

        # Remove DP/DDP 'module' prefix.
        consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")

        try:
            load_state_dict(out, state_dict)
        except (KeyError, ValueError) as ex:
            raise_error(ex)

        if model_device == META:
            # Non-persistent buffers are not included in the checkpoint, so we
            # have to explicitly initialize them.
            reset_non_persistent_buffers(out)

        self._root_gang.barrier()

    @override
    def load_last_model(self, out: Module, *, device: Optional[Device] = None) -> int:
        last_step_nr = self.get_last_step_number(with_model=True)
        if last_step_nr is None:
            raise CheckpointNotFoundError("No checkpoint found.")

        self.load_model(last_step_nr, out, device=device)

        return last_step_nr

    def get_model_path(self, step_nr: Optional[int] = None) -> Optional[Path]:
        """Return the path of the model of the specified training step.

        :param step_nr:
            The number of the training step. If ``None``, returns the path of
            the last model in the training.
        """
        if step_nr is None:
            step_nr = self.get_last_step_number(with_model=True)

        if step_nr is None:
            return None

        return self._checkpoint_dir.joinpath(
            f"step_{step_nr}/model{self._shard_suffix}.pt"
        )

    # compat
    def get_model_checkpoint_path(
        self, step_nr: Optional[int] = None
    ) -> Optional[Path]:
        return self.get_model_path(step_nr)

    @override
    def has_checkpoint(
        self, step_nr: Optional[int] = None, *, with_model: bool = False
    ) -> bool:
        it = self._iter_step_numbers(with_model)

        if step_nr is None:
            return next(it, None) is not None

        return step_nr in it

    @override
    def get_step_numbers(self, *, with_model: bool = False) -> List[int]:
        step_numbers = list(self._iter_step_numbers(with_model))

        step_numbers.sort()

        return step_numbers

    @override
    def get_last_step_number(self, *, with_model: bool = False) -> Optional[int]:
        if step_numbers := self.get_step_numbers(with_model=with_model):
            return step_numbers[-1]

        return None

    def _iter_step_numbers(self, with_model: bool) -> Iterator[int]:
        try:
            for step_dir in self._checkpoint_dir.glob("step_*"):
                if not step_dir.is_dir():
                    continue

                try:
                    step_nr = int(step_dir.name[5:])
                except ValueError:
                    continue

                if with_model:
                    if self._distributed_fs:
                        # On NFS, `exists()` might return a stale answer for
                        # cached LOOKUP results.
                        self._clear_nfs_lookup_cache(step_dir)

                    if not step_dir.joinpath(f"model{self._shard_suffix}.pt").exists():
                        continue

                yield step_nr
        except OSError as ex:
            raise RuntimeError(
                "The base checkpoint directory cannot be traversed. See nested exception for details."
            ) from ex

    @staticmethod
    def _clear_nfs_lookup_cache(path: Path) -> None:
        # Use the `opendir`/`readdir`/`closedir` trick to drop all cached NFS
        # LOOKUP results for `path`.
        try:
            it = scandir(path)
        except FileNotFoundError:
            return

        try:
            next(it)
        except StopIteration:
            pass
        finally:
            it.close()


class CheckpointNotFoundError(RuntimeError):
    """Raised when a checkpoint is not found."""
