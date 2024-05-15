# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from contextlib import nullcontext
from os import scandir
from pathlib import Path
from pickle import PickleError
from shutil import rmtree
from typing import (
    Any,
    ContextManager,
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
import yaml
from torch.distributed._shard import load_with_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.nn import Module

from fairseq2.gang import Gang
from fairseq2.models.utils.checkpoint import load_checkpoint
from fairseq2.typing import CPU, DataClass, override
from fairseq2.utils.dataclass import to_safe_dict


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
    _num_shards: int
    _shard_suffix: str
    _model_key: str
    _replicated_keys: Set[str]

    def __init__(
        self,
        checkpoint_dir: Path,
        gang: Gang,
        *,
        dp_gang: Optional[Gang] = None,
        tp_gang: Optional[Gang] = None,
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
        :param model_key:
            The key of the model in provided checkpoints.
        :param replicated_keys:
            The keys in provided checkpoints whose values are replicated across
            all processes in the gang.
        """
        self._checkpoint_dir = checkpoint_dir

        self._root_gang = gang

        self._dp_gang = gang

        self._num_shards = 1

        self._shard_suffix = ""

        if dp_gang is not None and tp_gang is not None:
            self._dp_gang = dp_gang

            self._num_shards = tp_gang.size

            if self._num_shards > 1:
                self._shard_suffix = f".{tp_gang.rank}"
        elif dp_gang is not None or tp_gang is not None:
            raise ValueError("`dp_gang` and `tp_gang` must be both specified.")

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

    def set_model_metadata(
        self,
        *,
        family: Optional[str] = None,
        base: Optional[str] = None,
        config: Optional[DataClass] = None,
    ) -> None:
        """Set the model metadata.

        :param family:
            The family of the model.
        :param base:
            The model that the model is based on.
        :param config:
            The configuration of the model.
        """
        if self._root_gang.rank == 0:
            metadata: Dict[str, Any] = {"name": "checkpoint"}

            if base is not None:
                metadata["base"] = base

            if family is not None:
                metadata["model_family"] = family

            if config is not None:
                metadata["model_config"] = to_safe_dict(config)

            if self._num_shards != 1:
                metadata["num_shards"] = f"{self._num_shards}"

            if self._num_shards == 1:
                filename = "model.pt"
            else:
                filename = "model.{shard_idx}.pt"

            last_step_metadata = {
                "base": "checkpoint",
                "name": "last_checkpoint",
                "checkpoint": f"last_step/{filename}",
            }

            def raise_error(cause: Exception) -> NoReturn:
                raise RuntimeError(
                    "The model metadata cannot be set. See nested exception for details."
                ) from cause

            try:
                self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            except OSError as ex:
                raise_error(ex)

            metadata_file = self._checkpoint_dir.joinpath("model.yaml")

            try:
                fp = metadata_file.open("w")
            except OSError as ex:
                raise_error(ex)

            try:
                yaml.safe_dump_all([metadata, last_step_metadata], fp)
            except OSError as ex:
                raise_error(ex)
            finally:
                fp.close()

        self._root_gang.barrier()

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

        # Create the checkpoint directory.
        if self._root_gang.rank == 0:
            try:
                tmp_step_dir.mkdir(parents=True)
            except OSError as ex:
                raise_error(ex)

        self._root_gang.barrier()

        # Do not modify the argument in-place. In case we fail, it should stay
        # intact.
        rank_part = checkpoint.copy()

        # Save the model.
        if self._model_replicated():
            if (state_dict := rank_part.pop(self._model_key, None)) is not None:
                if self._dp_gang.rank == 0:
                    model_file = tmp_step_dir.joinpath(f"model{self._shard_suffix}.pt")

                    try:
                        torch.save({"model": state_dict}, model_file)
                    except (RuntimeError, OSError, PickleError) as ex:
                        raise_error(ex)

                self._root_gang.barrier()

        # Save the replicated state.
        if self._replicated_keys:
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

        # Save the per-rank state.
        if not skip_rank:
            rank_file = tmp_step_dir.joinpath(
                f"rank_{self._dp_gang.rank}{self._shard_suffix}.pt"
            )

            try:
                torch.save(rank_part, rank_file)
            except (RuntimeError, OSError, PickleError) as ex:
                raise_error(ex)

            self._root_gang.barrier()

        # Save the checkpoint metadata.
        if metadata is not None:
            if self._dp_gang.rank == 0:
                metadata_file = tmp_step_dir.joinpath(
                    f"metadata{self._shard_suffix}.pt"
                )

                try:
                    torch.save(metadata, metadata_file)
                except (RuntimeError, OSError, PickleError) as ex:
                    raise_error(ex)

            self._root_gang.barrier()

        # Save the model metadata.
        if self._root_gang.rank == 0:
            if self._num_shards == 1:
                model_filename = "model.pt"
            else:
                model_filename = "model.{shard_idx}.pt"

            model_metadata = {
                "base": "checkpoint",
                "name": f"checkpoint_step_{step_nr}",
                "checkpoint": model_filename,
            }

            model_metadata_file = tmp_step_dir.joinpath("model.yaml")

            try:
                fp = model_metadata_file.open("w")
            except OSError as ex:
                raise_error(ex)

            try:
                yaml.safe_dump(model_metadata, fp)
            except OSError as ex:
                raise_error(ex)
            finally:
                fp.close()

        # Commit the checkpoint.
        if self._root_gang.rank == 0:
            step_dir = tmp_step_dir.with_suffix("")

            try:
                tmp_step_dir.replace(step_dir)
            except OSError as ex:
                raise_error(ex)

            tmp_link = self._checkpoint_dir.joinpath("last_step.tmp")

            try:
                tmp_link.unlink(missing_ok=True)

                tmp_link.symlink_to(step_dir.name, target_is_directory=True)

                link = tmp_link.with_suffix("")

                tmp_link.replace(link)
            except OSError as ex:
                raise_error(ex)

        self._root_gang.barrier()

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

        def maybe_with_dp_process_group() -> ContextManager[None]:
            try:
                pg = self._dp_gang.as_process_group()
            except RuntimeError:
                return nullcontext()

            return load_with_process_group(pg)

        # Load PyTorch's `ShardedTensor`s with the right gang.
        with maybe_with_dp_process_group():
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
        if self._root_gang.rank == 0:
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

        self._root_gang.barrier()

        for step_number in step_numbers[:-n]:
            self.delete_checkpoint(step_number)

    # compat
    def get_model_checkpoint_path(
        self, step_nr: Optional[int] = None
    ) -> Optional[Path]:
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

    # compat
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
                    # On NFS, `exists()` might return a stale answer for cached
                    # LOOKUP results.
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
