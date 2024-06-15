# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from pickle import PickleError
from shutil import rmtree
from typing import (
    AbstractSet,
    Any,
    ContextManager,
    Dict,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Optional,
    Set,
    Tuple,
    final,
)

import yaml
from torch.distributed._shard import load_with_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.nn import Module

from fairseq2.assets.metadata_provider import (
    AbstractAssetMetadataProvider,
    AssetMetadataError,
    _load_metadata_file,
)
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.typing import CPU, DataClass, override
from fairseq2.utils.dataclass import to_safe_dict
from fairseq2.utils.file import TensorDumper, TensorLoader, dump_tensors, load_tensors

log = get_log_writer(__name__)


class CheckpointManager(ABC):
    """Saves and loads training checkpoints."""

    @abstractmethod
    def save_checkpoint(
        self,
        step_nr: int,
        checkpoint: Mapping[str, Any],
        *,
        model_key: str = "model",
        replicated_keys: Optional[AbstractSet[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save the checkpoint of the specified training step.

        :param step_nr:
            The number of the training step.
        :param checkpoint:
            The checkpoint to save.
        :param model_key:
            The key of the model in ``checkpoint``.
        :param replicated_keys:
            The keys in ``checkpoint`` whose values are replicated across all
            processes in the gang.
        :param metadata:
            The checkpoint metadata to save. Must be pickeable.
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
    def has_checkpoint(self, step_nr: Optional[int] = None) -> bool:
        """Return ``True`` if the manager holds a checkpoint.

        :param step_nr:
            The number of the training step. If ``None``, returns ``True`` if
            the manager holds at least one checkpoint.
        """

    @abstractmethod
    def get_step_numbers(self) -> List[int]:
        """Return the numbers of the training steps that have a checkpoint."""

    # compat
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
    _tensor_loader: TensorLoader
    _tensor_dumper: TensorDumper

    # compat
    replicated_keys: Set[str]

    def __init__(
        self,
        checkpoint_dir: Path,
        gang: Gang,
        *,
        dp_gang: Optional[Gang] = None,
        tp_gang: Optional[Gang] = None,
        tensor_loader: Optional[TensorLoader] = None,
        tensor_dumper: Optional[TensorDumper] = None,
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
        :param tensor_loader:
            The tensor loader to load checkpoints into memory.
        :param tensor_dumper:
            The tensor dumper to save checkpoints into file.
        """
        self._checkpoint_dir = checkpoint_dir.expanduser().resolve()

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

        self._tensor_loader = tensor_loader or load_tensors
        self._tensor_dumper = tensor_dumper or dump_tensors

        # compat
        self.replicated_keys = set()

    def save_model_metadata(
        self,
        *,
        base_asset: Optional[str] = None,
        family: Optional[str] = None,
        config: Optional[DataClass] = None,
    ) -> None:
        """Set the model metadata.

        :param base_asset:
            The asset that the model is based on.
        :param family:
            The family of the model.
        :param config:
            The configuration of the model.
        """
        if self._root_gang.rank == 0:
            metadata: Dict[str, Any] = {"name": "checkpoint"}

            if base_asset is not None:
                metadata["base"] = base_asset

            if family is not None:
                metadata["model_family"] = family

            if config is not None:
                metadata["model_config"] = to_safe_dict(config)

            if self._num_shards != 1:
                metadata["num_shards"] = self._num_shards

            def raise_error(cause: Exception) -> NoReturn:
                raise RuntimeError(
                    "The model metadata cannot be saved. See nested exception for details."
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
                yaml.safe_dump(metadata, fp, sort_keys=False)
            except OSError as ex:
                raise_error(ex)
            finally:
                fp.close()

        self._root_gang.barrier()

    @override
    def save_checkpoint(
        self,
        step_nr: int,
        checkpoint: Mapping[str, Any],
        *,
        model_key: str = "model",
        replicated_keys: Optional[AbstractSet[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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

        # Do not modify `checkpoint` in-place. In case we fail, it should stay
        # intact.
        rank_part = dict(checkpoint)

        rank_part["model_key"] = model_key

        # compat
        if replicated_keys is None:
            replicated_keys = self.replicated_keys

        def model_replicated() -> bool:
            if self._dp_gang.size == 1:
                return True

            if not replicated_keys:
                return False

            return model_key in replicated_keys or "*" in replicated_keys

        # Save the model.
        if model_replicated():
            if (state_dict := rank_part.pop(model_key, None)) is not None:
                del rank_part["model_key"]

                if self._dp_gang.rank == 0:
                    model_file = tmp_step_dir.joinpath(f"model{self._shard_suffix}.pt")

                    try:
                        self._tensor_dumper(
                            {model_key: state_dict, "model_key": model_key}, model_file
                        )
                    except (RuntimeError, OSError, PickleError) as ex:
                        raise_error(ex)

                self._root_gang.barrier()

        # Save the replicated state.
        if replicated_keys:
            if self._dp_gang.rank == 0:
                replicated_part = {}

                if "*" in replicated_keys:
                    replicated_part, rank_part = rank_part, replicated_part
                else:
                    for key in replicated_keys:
                        try:
                            replicated_part[key] = rank_part.pop(key)
                        except KeyError:
                            pass

                if replicated_part:
                    replicated_file = tmp_step_dir.joinpath(
                        f"replicated{self._shard_suffix}.pt"
                    )

                    try:
                        self._tensor_dumper(replicated_part, replicated_file)
                    except (RuntimeError, OSError, PickleError) as ex:
                        raise_error(ex)
            else:
                if "*" in replicated_keys:
                    rank_part.clear()
                else:
                    for key in replicated_keys:
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
                self._tensor_dumper(rank_part, rank_file)
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
                    self._tensor_dumper(metadata, metadata_file)
                except (RuntimeError, OSError, PickleError) as ex:
                    raise_error(ex)

            self._root_gang.barrier()

        # Commit the checkpoint.
        if self._root_gang.rank == 0:
            step_dir = tmp_step_dir.with_suffix("")

            try:
                tmp_step_dir.replace(step_dir)
            except OSError as ex:
                raise_error(ex)

            # Update the 'last_step' symbolic link.
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
        log.info("Extracting consolidated FSDP state dictionary of the model.")

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()

        self._root_gang.barrier()

        log.info("Model state dictionary extracted. Saving to file on data parallel rank 0 (per shard).")  # fmt: skip

        if self._dp_gang.rank == 0:
            tmp_model_file = self._checkpoint_dir.joinpath(
                f"step_{step_nr}/model{self._shard_suffix}.tmp"
            )

            if not tmp_model_file.parent.exists():
                raise RuntimeError(
                    f"A consolidated FSDP model can only be saved for training steps that have a checkpoint, but training step {step_nr} has no checkpoint."
                )

            try:
                self._tensor_dumper(
                    {"model": state_dict, "model_key": "model"}, tmp_model_file
                )
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

        def maybe_with_dp_process_group() -> ContextManager[None]:
            try:
                pg = self._dp_gang.as_process_group()
            except RuntimeError:
                return nullcontext()

            return load_with_process_group(pg)

        step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

        def load_part(filename: str) -> Dict[str, Any]:
            with maybe_with_dp_process_group():  # Required for `ShardedTensor`.
                try:
                    part = self._tensor_loader(
                        step_dir.joinpath(filename), map_location=CPU
                    )
                except FileNotFoundError:
                    part = {}
                except (RuntimeError, OSError, PickleError) as ex:
                    raise_error(ex)

                self._root_gang.barrier()

                return part

        checkpoint = {}

        suffix = self._shard_suffix

        for f in [f"rank_{self._dp_gang.rank}{suffix}.pt", f"replicated{suffix}.pt"]:
            part = load_part(f)

            # Consolidate the checkpoint parts.
            checkpoint.update(part)

        try:
            model_key = checkpoint["model_key"]
        except KeyError:
            model_key = None

        # If we don't have the model state in the checkpoint so far, it means it
        # was replicated.
        if model_key not in checkpoint:
            part = load_part(f"model{self._shard_suffix}.pt")

            checkpoint.update(part)

        try:
            del checkpoint["model_key"]
        except KeyError:
            pass

        if not checkpoint:
            raise CheckpointNotFoundError(f"Training step {step_nr} has no checkpoint.")

        return checkpoint

    @override
    def load_last_checkpoint(self) -> Tuple[int, Dict[str, Any]]:
        step_numbers = self.get_step_numbers()
        if not step_numbers:
            raise CheckpointNotFoundError("No checkpoint found.")

        last_step_nr = step_numbers[-1]

        checkpoint = self.load_checkpoint(last_step_nr)

        return last_step_nr, checkpoint

    @override
    def load_metadata(self, step_nr: int) -> Optional[Dict[str, Any]]:
        metadata_file = self._checkpoint_dir.joinpath(
            f"step_{step_nr}/metadata{self._shard_suffix}.pt"
        )

        try:
            metadata = self._tensor_loader(metadata_file, map_location=CPU)
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
    def has_checkpoint(self, step_nr: Optional[int] = None) -> bool:
        it = self._iter_step_numbers()

        if step_nr is None:
            return next(it, None) is not None

        return step_nr in it

    @override
    def get_step_numbers(self) -> List[int]:
        step_numbers = list(self._iter_step_numbers())

        step_numbers.sort()

        return step_numbers

    # compat
    @override
    def get_last_step_number(self, *, with_model: bool = False) -> Optional[int]:
        if step_numbers := self.get_step_numbers():
            return step_numbers[-1]

        return None

    def _iter_step_numbers(self) -> Iterator[int]:
        try:
            for step_dir in self._checkpoint_dir.glob("step_*"):
                if not step_dir.is_dir():
                    continue

                try:
                    step_nr = int(step_dir.name[5:])
                except ValueError:
                    continue

                yield step_nr
        except OSError as ex:
            raise RuntimeError(
                "The base checkpoint directory cannot be traversed. See nested exception for details."
            ) from ex


class CheckpointNotFoundError(RuntimeError):
    """Raised when a checkpoint is not found."""


@final
class CheckpointModelMetadataProvider(AbstractAssetMetadataProvider):
    """Provides checkpoint model metadata saved by a :class:`FileCheckpointManager.`"""

    _checkpoint_dir: Path

    def __init__(self, checkpoint_dir: Path) -> None:
        super().__init__()

        self._checkpoint_dir = checkpoint_dir.expanduser().resolve()

    @override
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        metadata_file = self._checkpoint_dir.joinpath("model.yaml")
        if not metadata_file.exists():
            raise AssetMetadataError(
                "The checkpoint model metadata cannot be found. Make sure to call `FileCheckpointManager.save_model_metadata()` first."
            )

        cache = dict(_load_metadata_file(metadata_file))

        try:
            metadata = cache["checkpoint@"]
        except KeyError as ex:
            raise AssetMetadataError(
                "The checkpoint model metadata has an invalid format."
            ) from ex

        try:
            num_shards = int(metadata["num_shards"])
        except KeyError:
            num_shards = 1
        except ValueError as ex:
            raise AssetMetadataError(
                "The checkpoint model metadata has an invalid format."
            ) from ex

        if num_shards == 1:
            filename = "model.pt"
        else:
            filename = "model.{shard_idx}.pt"

        def add_checkpoint_metadata(name: str, path: Path) -> None:
            cache[name] = {"base": "checkpoint", "checkpoint": str(path)}

        add_checkpoint_metadata(
            "last_checkpoint@", self._checkpoint_dir.joinpath(f"last_step/{filename}")
        )

        try:
            for step_dir in self._checkpoint_dir.glob("step_*"):
                if not step_dir.is_dir():
                    continue

                try:
                    step_nr = int(step_dir.name[5:])
                except ValueError:
                    continue

                add_checkpoint_metadata(
                    f"checkpoint_step_{step_nr}@", step_dir.joinpath(filename)
                )

        except OSError as ex:
            raise RuntimeError(
                "The base checkpoint directory cannot be traversed. See nested exception for details."
            ) from ex

        return cache
