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
from typing import Any, Dict, Iterator, List, Optional, Tuple, final

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.nn import Module

from fairseq2.gang import Gang
from fairseq2.models.utils.checkpoint import load_checkpoint
from fairseq2.nn.utils.module import (
    infer_device,
    reset_non_persistent_buffers,
    to_empty,
)
from fairseq2.typing import CPU, META, Device, finaloverride


class CheckpointManager(ABC):
    """Saves and loads training checkpoints."""

    @abstractmethod
    def save_checkpoint(
        self,
        step_nr: int,
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Save the checkpoint of the specified training step.

        :param step_nr:
            The number of the training step.
        :param checkpoint:
            The checkpoint to save.
        :param metadata:
            The metadata (e.g. training config) associated with the checkpoint.
        """

    @abstractmethod
    def load_checkpoint(
        self, step_nr: int
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Load the checkpoint of the specified training step.

        :param step_nr:
            The number of the training step.

        :returns:
            - The checkpoint
            - The metadata associated with the checkpoint.
        """

    @abstractmethod
    def load_last_checkpoint(
        self,
    ) -> Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Load the last checkpoint in the training.

        :returns:
            - The number of the training step.
            - The checkpoint.
            - The metadata associated with the checkpoint.
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
    def load_consolidated_model(
        self, step_nr: int, out: Module, device: Optional[Device] = None
    ) -> None:
        """Load the consolidated model at the specified training step.

        :param step_nr:
            The number of the training step.
        :param out:
            The output model to load.
        :param device:
            The device on which to load ``out`` if it is on the meta device.
        """

    @abstractmethod
    def load_last_consolidated_model(
        self, out: Module, device: Optional[Device] = None
    ) -> int:
        """Load the last consolidated model in the training.

        :param out:
            The output model to load.
        :param device:
            The device on which to load ``out`` if it is on the meta device.

        :returns:
            The number of the training step associated with the consolidated
            model.
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
            If ``True``, only considers checkpoints with a consolidated model.
        """

    @abstractmethod
    def get_step_numbers(self, *, with_model: bool = False) -> List[int]:
        """Return the numbers of the training steps that have a checkpoint.

        :param with_model:
            If ``True``, only considers checkpoints with a consolidated model.
        """

    @abstractmethod
    def get_last_step_number(self, *, with_model: bool = False) -> Optional[int]:
        """Return the number of the training step associated with the last
        checkpoint.

        :param with_model:
            If ``True``, only considers checkpoints with a consolidated model.
        """


@final
class FileCheckpointManager(CheckpointManager):
    """Saves and loads training checkpoints on a file system."""

    checkpoint_dir: Path
    gang: Gang
    distributed_fs: bool
    replicated_keys: Optional[List[str]]

    def __init__(
        self, checkpoint_dir: Path, gang: Gang, distributed_fs: bool = True
    ) -> None:
        """
        :param checkpoint_dir:
            The root directory under which to store the checkpoints.
        :param gang:
            The gang to coordinate the checkpoint operations.
        :param distributed_fs:
            If ``True``, the underlying file system of ``checkpoint_dir`` is
            considered distributed (e.g. NFS).
        """
        self.gang = gang
        self.distributed_fs = distributed_fs

        if distributed_fs:
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = checkpoint_dir.joinpath(f"rank_{self.gang.rank}")

        self.replicated_keys = None

    @finaloverride
    def save_checkpoint(
        self,
        step_nr: int,
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if "metadata" in checkpoint:
            raise ValueError(
                "`checkpoint` must not include the reserved 'metadata' key."
            )

        self.delete_checkpoint(step_nr, missing_ok=True)

        tmp_step_dir = self.checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

        if self.gang.rank == 0 or not self.distributed_fs:
            try:
                tmp_step_dir.mkdir(parents=True)
            except OSError as ex:
                raise RuntimeError(
                    f"The checkpoint directory for training step {step_nr} cannot be created. See nested exception for details."
                ) from ex

        self.gang.barrier()

        rank_checkpoint = checkpoint.copy()

        if metadata is not None:
            rank_checkpoint["metadata"] = metadata

        # For non-distributed file systems, we disregard the replicated keys and
        # force each process in the gang to save the full checkpoint.
        if (keys := self.replicated_keys) and self.distributed_fs:
            full_replica = len(keys) == 1 and keys[0] == "*"

            if self.gang.rank == 0:
                replicated_checkpoint = {}

                if full_replica:
                    replicated_checkpoint, rank_checkpoint = rank_checkpoint, replicated_checkpoint  # fmt: skip
                else:
                    for key in keys:
                        try:
                            replicated_checkpoint[key] = rank_checkpoint.pop(key)
                        except KeyError:
                            pass

                if replicated_checkpoint:
                    checkpoint_file = tmp_step_dir.joinpath("replicated.pt")

                    try:
                        torch.save(replicated_checkpoint, checkpoint_file)
                    except (RuntimeError, OSError, PickleError) as ex:
                        raise RuntimeError(
                            f"The checkpoint of training step {step_nr} cannot be saved. See nested exception for details."
                        ) from ex
            else:
                if full_replica:
                    rank_checkpoint.clear()
                else:
                    for key in keys:
                        try:
                            del rank_checkpoint[key]
                        except KeyError:
                            pass

            self.gang.barrier()

            # Check if anything is left to save in the rank checkpoint.
            skip_rank = not rank_checkpoint
        else:
            skip_rank = False

        if not skip_rank:
            checkpoint_file = tmp_step_dir.joinpath(f"rank_{self.gang.rank}.pt")

            try:
                torch.save(rank_checkpoint, checkpoint_file)
            except (RuntimeError, OSError, PickleError) as ex:
                raise RuntimeError(
                    f"The checkpoint of training step {step_nr} cannot be saved. See nested exception for details."
                ) from ex

            self.gang.barrier()

        if self.gang.rank == 0 or not self.distributed_fs:
            try:
                tmp_step_dir.replace(tmp_step_dir.with_suffix(""))
            except OSError as ex:
                raise RuntimeError(
                    f"The checkpoint directory for training step {step_nr} cannot be renamed. See nested exception for details."
                ) from ex

        self.gang.barrier()

    @finaloverride
    def load_checkpoint(
        self, step_nr: int
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        checkpoint_file = self.checkpoint_dir.joinpath(f"step_{step_nr}/replicated.pt")

        try:
            replicated_checkpoint = load_checkpoint(checkpoint_file, map_location=CPU)
        except FileNotFoundError:
            replicated_checkpoint = None
        except (RuntimeError, OSError, PickleError) as ex:
            raise RuntimeError(
                f"The checkpoint of training step {step_nr} cannot be loaded. See nested exception for details."
            ) from ex

        self.gang.barrier()

        checkpoint_file = self.checkpoint_dir.joinpath(
            f"step_{step_nr}/rank_{self.gang.rank}.pt"
        )

        try:
            rank_checkpoint = load_checkpoint(checkpoint_file, map_location=CPU)
        except FileNotFoundError:
            rank_checkpoint = None
        except (RuntimeError, OSError, PickleError) as ex:
            raise RuntimeError(
                f"The checkpoint of training step {step_nr} cannot be loaded. See nested exception for details."
            ) from ex

        self.gang.barrier()

        if replicated_checkpoint is None:
            if rank_checkpoint is None:
                raise CheckpointNotFoundError(
                    f"Training step {step_nr} has no checkpoint."
                )

            checkpoint = rank_checkpoint
        else:
            if rank_checkpoint is not None:
                replicated_checkpoint.update(rank_checkpoint)

            checkpoint = replicated_checkpoint

        metadata = checkpoint.pop("metadata", None)

        return checkpoint, metadata

    @finaloverride
    def load_last_checkpoint(
        self,
    ) -> Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]:
        last_step_nr = self.get_last_step_number()
        if last_step_nr is None:
            raise CheckpointNotFoundError("No checkpoint can be found.")

        # If we don't have a distributed file system, we have to ensure that we
        # have a consistent view of checkpoints across all processes.
        if not self.distributed_fs:
            step_numbers = torch.empty(
                (self.gang.size,), device=self.gang.device, dtype=torch.int64
            )

            self.gang.all_gather(
                step_numbers, torch.tensor(last_step_nr, device=self.gang.device)
            )

            if not (step_numbers == last_step_nr).all():
                raise RuntimeError(
                    f"The processes in the gang have no consensus on the last training step. The last step numbers sorted by rank: {step_numbers.tolist()}"
                )

        checkpoint, metadata = self.load_checkpoint(last_step_nr)

        return last_step_nr, checkpoint, metadata

    @finaloverride
    def delete_checkpoint(self, step_nr: int, *, missing_ok: bool = False) -> None:
        if self.gang.rank == 0 or not self.distributed_fs:
            step_dir = self.checkpoint_dir.joinpath(f"step_{step_nr}")

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

        self.gang.barrier()

    @finaloverride
    def keep_last_n_checkpoints(self, n: int) -> None:
        step_numbers = self.get_step_numbers()

        for step_number in step_numbers[:-n]:
            self.delete_checkpoint(step_number)

    @finaloverride
    def save_consolidated_model(self, step_nr: int, model: Module) -> None:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()

        checkpoint = {"model": state_dict}

        if self.gang.rank == 0:
            step_dir = self.checkpoint_dir.joinpath(f"step_{step_nr}")

            try:
                step_dir.mkdir(parents=True, exist_ok=True)
            except OSError as ex:
                raise RuntimeError(
                    f"The checkpoint directory of training step {step_nr} cannot be created. See nested exception for details."
                ) from ex

            tmp_model_file = step_dir.joinpath("model.pt.tmp")

            try:
                torch.save(checkpoint, tmp_model_file)
            except (RuntimeError, OSError, PickleError) as ex:
                raise RuntimeError(
                    f"The consolidated model of training step {step_nr} cannot be saved. See nested exception for details."
                ) from ex

            try:
                tmp_model_file.replace(step_dir.joinpath("model.pt"))
            except OSError as ex:
                raise RuntimeError(
                    f"The consolidated model file of training step {step_nr} cannot be renamed. See nested exception for details."
                ) from ex

        self.gang.barrier()

    @finaloverride
    def load_consolidated_model(
        self, step_nr: int, out: Module, device: Optional[Device] = None
    ) -> None:
        if self.gang.rank == 0:
            model_file = self.checkpoint_dir.joinpath(f"step_{step_nr}/model.pt")

            try:
                checkpoint = load_checkpoint(
                    model_file, map_location=CPU, restrict=True
                )
            except FileNotFoundError:
                raise CheckpointNotFoundError(
                    f"Training step {step_nr} has no consolidated model."
                )
            except (RuntimeError, OSError, PickleError) as ex:
                raise RuntimeError(
                    f"The consolidated model checkpoint of training step {step_nr} cannot be loaded. See nested exception for details."
                ) from ex

            model_device = infer_device(out)

            if model_device == META:
                # Move the model to the actual device without initializing. Its
                # state will be overwritten by the checkpoint anyways.
                to_empty(out, device=device or CPU)

            # Load the model.
            try:
                state_dict = checkpoint["model"]
            except KeyError:
                raise RuntimeError(
                    f"The consolidated model checkpoint of training step {step_nr} does not contain a 'model' entry."
                )

            try:
                out.load_state_dict(state_dict)
            except (KeyError, ValueError) as ex:
                raise RuntimeError(
                    f"The consolidated model of training step {step_nr} cannot be loaded. See nested exception for details."
                ) from ex

            if model_device == META:
                # Non-persistent buffers are not included in the checkpoint, so
                # we have to explicitly initialize them.
                reset_non_persistent_buffers(out)

        self.gang.barrier()

    @finaloverride
    def load_last_consolidated_model(
        self, out: Module, device: Optional[Device] = None
    ) -> int:
        if self.gang.rank == 0:
            last_step_nr = self.get_last_step_number(with_model=True)
            if last_step_nr is None:
                raise CheckpointNotFoundError("No checkpoint can be found.")
        else:
            last_step_nr = 0

        self.load_consolidated_model(last_step_nr, out, device)

        return last_step_nr

    @finaloverride
    def has_checkpoint(
        self, step_nr: Optional[int] = None, *, with_model: bool = False
    ) -> bool:
        it = self._iter_step_numbers(with_model)

        if step_nr is None:
            return next(it, None) is not None

        return step_nr in it

    @finaloverride
    def get_step_numbers(self, *, with_model: bool = False) -> List[int]:
        step_numbers = list(self._iter_step_numbers(with_model))

        step_numbers.sort()

        return step_numbers

    @finaloverride
    def get_last_step_number(self, *, with_model: bool = False) -> Optional[int]:
        if step_numbers := self.get_step_numbers(with_model=with_model):
            return step_numbers[-1]

        return None

    def _iter_step_numbers(self, with_model: bool) -> Iterator[int]:
        try:
            for step_dir in self.checkpoint_dir.glob("step_*"):
                if not step_dir.is_dir():
                    continue

                try:
                    step_nr = int(step_dir.name[5:])
                except ValueError:
                    continue

                if with_model:
                    if self.distributed_fs:
                        # On NFS, `exists()` might return a stale answer for cached
                        # LOOKUP results.
                        self._clear_nfs_lookup_cache(step_dir)

                    if not step_dir.joinpath("model.pt").exists():
                        continue

                yield step_nr
        except OSError as ex:
            raise RuntimeError(
                "The root checkpoint directory cannot be traversed. See nested exception for details."
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
