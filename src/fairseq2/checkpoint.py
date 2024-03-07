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
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, final

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.nn import Module

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
            The checkpoint metadata (e.g. training config) to save.
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
    def save_consolidated_model(self, step_nr: int, model: Module) -> None:
        """Save ``model`` with a ``state_dict`` consolidated from all processes.

        :param step_nr:
            The number of the training step.
        :param model:
            The model to save.
        """

    @abstractmethod
    def load_model(
        self, step_nr: int, out: Module, *, device: Optional[Device] = None
    ) -> None:
        """Load the model of the specified training step.

        :param step_nr:
            The number of the training step.
        :param out:
            The output model to load.
        :param device:
            The device on which to load ``out`` if it is on the meta device.
        """

    @abstractmethod
    def load_last_model(self, out: Module, *, device: Optional[Device] = None) -> int:
        """Load the last saved model in the training.

        :param out:
            The output model to load.
        :param device:
            The device on which to load ``out`` if it is on the meta device.

        :returns:
            The number of the training step of the model.
        """

    @abstractmethod
    def has_checkpoint(
        self, step_nr: Optional[int] = None, *, with_model: bool = False
    ) -> bool:
        """Return ``True`` if the manager holds a checkpoint.

        :param step_nr:
            If not ``None``, returns ``True`` if the manager holds the
            checkpoint of the specified training step; otherwise, returns
            ``True`` if the manager holds at least one checkpoint.
        :param with_model:
            If ``True``, only considers checkpoints with a model.
        """

    @abstractmethod
    def get_step_numbers(self, *, with_model: bool = False) -> List[int]:
        """Return the numbers of the training steps that have a checkpoint.

        :param with_model:
            If ``True``, only considers checkpoints with a model.
        """

    @abstractmethod
    def get_last_step_number(self, *, with_model: bool = False) -> Optional[int]:
        """Return the number of the training step of the last checkpoint.

        :param with_model:
            If ``True``, only considers checkpoints with a model.
        """


@final
class FileCheckpointManager(CheckpointManager):
    """Saves and loads training checkpoints on a file system."""

    _checkpoint_dir: Path
    _gang: Gang
    _distributed_fs: bool

    replicated_keys: Set[str]

    def __init__(
        self, checkpoint_dir: Path, gang: Gang, *, distributed_fs: bool = True
    ) -> None:
        """
        :param checkpoint_dir:
            The base directory under which to store the checkpoints.
        :param gang:
            The gang to coordinate the checkpoint operations.
        :param distributed_fs:
            If ``True``, the underlying file system of ``checkpoint_dir`` is
            considered distributed (e.g. NFS).
        """
        self._gang = gang
        self._distributed_fs = distributed_fs

        if distributed_fs:
            self._checkpoint_dir = checkpoint_dir
        else:
            self._checkpoint_dir = checkpoint_dir.joinpath(f"rank_{self._gang.rank}")

        self.replicated_keys = set()

    def _is_replicated_key(self, key: str) -> bool:
        if self._save_full_replica():
            return True

        return key in self.replicated_keys

    def _save_full_replica(self) -> bool:
        return len(self.replicated_keys) == 1 and "*" in self.replicated_keys

    @override
    def save_checkpoint(
        self,
        step_nr: int,
        checkpoint: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        self.delete_checkpoint(step_nr, missing_ok=True)

        tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

        if self._gang.rank == 0 or not self._distributed_fs:
            try:
                tmp_step_dir.mkdir(parents=True)
            except OSError as ex:
                raise RuntimeError(
                    f"The checkpoint of training step {step_nr} cannot be saved. See nested exception for details."
                ) from ex

        self._gang.barrier()

        # If the model is replicated, we always save it into its own file.
        if self._is_replicated_key("model"):
            if (state_dict := checkpoint.pop("model", None)) is not None:
                if self._gang.rank == 0 or not self._distributed_fs:
                    model_file = tmp_step_dir.joinpath("model.pt")

                    try:
                        torch.save({"model": state_dict}, model_file)
                    except (RuntimeError, OSError, PickleError) as ex:
                        raise RuntimeError(
                            f"The checkpoint of training step {step_nr} cannot be saved. See nested exception for details."
                        ) from ex

                self._gang.barrier()

        rank_part = checkpoint.copy()

        # For non-distributed file systems, we disregard the replicated keys and
        # force each process in the gang to save the full checkpoint.
        if self.replicated_keys and self._distributed_fs:
            if self._gang.rank == 0:
                replicated_part = {}

                if self._save_full_replica():
                    replicated_part, rank_part = rank_part, replicated_part
                else:
                    for key in self.replicated_keys:
                        try:
                            replicated_part[key] = rank_part.pop(key)
                        except KeyError:
                            pass

                if replicated_part:
                    replicated_file = tmp_step_dir.joinpath("replicated.pt")

                    try:
                        torch.save(replicated_part, replicated_file)
                    except (RuntimeError, OSError, PickleError) as ex:
                        raise RuntimeError(
                            f"The checkpoint of training step {step_nr} cannot be saved. See nested exception for details."
                        ) from ex
            else:
                if self._save_full_replica():
                    rank_part.clear()
                else:
                    for key in self.replicated_keys:
                        try:
                            del rank_part[key]
                        except KeyError:
                            pass

            self._gang.barrier()

            # Check if anything is left to save for the rank.
            skip_rank = not rank_part
        else:
            skip_rank = False

        if not skip_rank:
            rank_file = tmp_step_dir.joinpath(f"rank_{self._gang.rank}.pt")

            try:
                torch.save(rank_part, rank_file)
            except (RuntimeError, OSError, PickleError) as ex:
                raise RuntimeError(
                    f"The checkpoint of training step {step_nr} cannot be saved. See nested exception for details."
                ) from ex

            self._gang.barrier()

        if metadata is not None:
            if self._gang.rank == 0 or not self._distributed_fs:
                metadata_file = tmp_step_dir.joinpath("metadata.pt")

                try:
                    torch.save(metadata, metadata_file)
                except (RuntimeError, OSError, PickleError) as ex:
                    raise RuntimeError(
                        f"The checkpoint of training step {step_nr} cannot be saved. See nested exception for details."
                    ) from ex

            self._gang.barrier()

        if self._gang.rank == 0 or not self._distributed_fs:
            try:
                tmp_step_dir.replace(tmp_step_dir.with_suffix(""))
            except OSError as ex:
                raise RuntimeError(
                    f"The checkpoint of training step {step_nr} cannot be saved. See nested exception for details."
                ) from ex

        self._gang.barrier()

    @override
    def load_checkpoint(self, step_nr: int) -> Dict[str, Any]:
        parts = []

        filenames = ["replicated.pt", f"rank_{self._gang.rank}.pt"]

        if self._is_replicated_key("model"):
            filenames.append("model.pt")

        step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

        for filename in filenames:
            try:
                part = load_checkpoint(
                    step_dir.joinpath(filename), map_location=CPU, mmap=True
                )
            except FileNotFoundError:
                part = None
            except (RuntimeError, OSError, PickleError) as ex:
                raise RuntimeError(
                    f"The checkpoint of training step {step_nr} cannot be loaded. See nested exception for details."
                ) from ex

            if part is not None:
                parts.append(part)

            self._gang.barrier()

        if not parts:
            raise CheckpointNotFoundError(f"Training step {step_nr} has no checkpoint.")

        checkpoint = parts[0]

        for part in parts[1:]:
            checkpoint.update(part)

        return checkpoint

    @override
    def load_last_checkpoint(self) -> Tuple[int, Dict[str, Any]]:
        last_step_nr = self.get_last_step_number()
        if last_step_nr is None:
            raise CheckpointNotFoundError("No checkpoint found.")

        # If we don't have a distributed file system, we have to ensure that we
        # have a consistent view of checkpoints across all processes.
        if not self._distributed_fs:
            step_numbers = torch.empty(
                (self._gang.size,), device=self._gang.device, dtype=torch.int64
            )

            self._gang.all_gather(
                step_numbers, torch.tensor(last_step_nr, device=self._gang.device)
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
        metadata_file = self._checkpoint_dir.joinpath(f"step_{step_nr}/metadata.pt")

        try:
            metadata = load_checkpoint(metadata_file, map_location=CPU, mmap=True)
        except FileNotFoundError:
            metadata = None
        except (RuntimeError, OSError, PickleError) as ex:
            raise RuntimeError(
                f"The checkpoint metadata of training step {step_nr} cannot be loaded. See nested exception for details."
            ) from ex

        self._gang.barrier()

        return metadata

    @override
    def delete_checkpoint(self, step_nr: int, *, missing_ok: bool = False) -> None:
        if self._gang.rank == 0 or not self._distributed_fs:
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

        self._gang.barrier()

    @override
    def keep_last_n_checkpoints(self, n: int) -> None:
        step_numbers = self.get_step_numbers()

        for step_number in step_numbers[:-n]:
            self.delete_checkpoint(step_number)

    @override
    def save_consolidated_model(self, step_nr: int, model: Module) -> None:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()

        if self._gang.rank == 0:
            tmp_model_file = self._checkpoint_dir.joinpath(f"step_{step_nr}/model.tmp")

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

        self._gang.barrier()

    @override
    def load_model(
        self, step_nr: int, out: Module, *, device: Optional[Device] = None
    ) -> None:
        model_file = self.get_model_checkpoint_path(step_nr=step_nr)
        if model_file is None:
            raise CheckpointNotFoundError(
                f"Training step {step_nr} has no saved model."
            )

        try:
            checkpoint = load_checkpoint(model_file, map_location=CPU, restrict=True)
        except FileNotFoundError:
            raise CheckpointNotFoundError(
                f"Training step {step_nr} has no saved model."
            )
        except (RuntimeError, OSError, PickleError) as ex:
            raise RuntimeError(
                f"The model of training step {step_nr} cannot be loaded. See nested exception for details."
            ) from ex

        model_device = infer_device(out, param_name="out")

        if model_device == META:
            # Move the model to the actual device without initializing. Its
            # state will be overwritten by the checkpoint anyways.
            to_empty(out, device=device or CPU)

        # Load the model.
        try:
            state_dict = checkpoint["model"]
        except KeyError:
            raise RuntimeError(
                f"The model of training step {step_nr} cannot be loaded. See nested exception for details."
            )

        try:
            load_state_dict(out, state_dict)
        except (KeyError, ValueError) as ex:
            raise RuntimeError(
                f"The model of training step {step_nr} cannot be loaded. See nested exception for details."
            ) from ex

        if model_device == META:
            # Non-persistent buffers are not included in the checkpoint, so we
            # have to explicitly initialize them.
            reset_non_persistent_buffers(out)

        self._gang.barrier()

    @override
    def load_last_model(self, out: Module, *, device: Optional[Device] = None) -> int:
        last_step_nr = self.get_last_step_number(with_model=True)
        if last_step_nr is None:
            raise CheckpointNotFoundError("No checkpoint found.")

        self.load_model(last_step_nr, out, device=device)

        return last_step_nr

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

    def get_model_checkpoint_path(
        self, step_nr: Optional[int] = None
    ) -> Optional[Path]:
        """
        Return the path for the model checkpoint (by default, the last one).
        :param step_nr:
            The step for which to load the model. If ``None``, then the last checkpoint is loaded.
        """
        if step_nr is None:
            step_nr = self.get_last_step_number(with_model=True)
        if step_nr is None:
            return None
        return self._checkpoint_dir.joinpath(f"step_{step_nr}/model.pt")

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

                    if not step_dir.joinpath("model.pt").exists():
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
