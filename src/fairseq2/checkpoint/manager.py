# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
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
    Tuple,
    final,
)
from warnings import catch_warnings

import yaml
from torch.distributed._shard import load_with_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.nn import Module

from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.typing import CPU, DataClass, override
from fairseq2.utils.dataclass import to_safe_dict
from fairseq2.utils.file import TensorDumper, TensorLoader, dump_tensors, load_tensors

log = get_log_writer(__name__)


class CheckpointManager(ABC):
    """Saves and loads training checkpoints."""

    @abstractmethod
    def begin_checkpoint(self, step_nr: int) -> None:
        """Begin a transactional checkpoint operation.

        :param step_nr:
            The number of the training step.
        """

    @abstractmethod
    def save_state(
        self,
        state: Mapping[str, Any],
        *,
        model_key: str = "model",
        replicated_keys: Optional[AbstractSet[str]] = None,
    ) -> None:
        """Save the training state.

        :param state:
            The state to save.
        :param model_key:
            The key of the model in ``state``.
        :param replicated_keys:
            The keys in ``state`` whose values are replicated across all
            processes in the gang.
        """

    @abstractmethod
    def save_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Save ``metadata`` associated with the checkpoint.

        :param metadata:
            The metadata to save. Must be pickeable.
        """

    @abstractmethod
    def save_score(self, score: Optional[float]) -> None:
        """Save the score of the checkpoint."""

    @abstractmethod
    def save_consolidated_fsdp_model(self, model: Module) -> None:
        """Save ``model`` with a ``state_dict`` consolidated from all processes."""

    @abstractmethod
    def commit_checkpoint(self) -> None:
        """Commit the checkpoint after which it will be considered saved."""

    @abstractmethod
    def load_checkpoint(self, step_nr: int) -> Dict[str, Any]:
        """Load the checkpoint of the specified training step."""

    @abstractmethod
    def load_last_checkpoint(self) -> Tuple[int, Dict[str, Any]]:
        """Load the last checkpoint in the training.

        :returns:
            - The number of the training step.
            - The checkpoint.
        """

    @abstractmethod
    def load_metadata(self, step_nr: int) -> Optional[Dict[str, Any]]:
        """Load the checkpoint metadata of the specified training step."""

    @abstractmethod
    def delete_checkpoint(
        self, step_nr: int, *, missing_ok: bool = False, preserve_model: bool = False
    ) -> None:
        """Delete the checkpoint of the specified training step.

        :param step_nr:
            The number of the training step.
        :param missing_ok:
            If ``True``, does not raise error if the checkpoint does not exists.
        :param preserve_model:
            If ``True``, model won't be deleted.
        """

    @abstractmethod
    def keep_last_n_checkpoints(self, n: int, *, preserve_model: bool = False) -> None:
        """Delete all but the last ``n`` checkpoints.

        :param n:
            The number of checkpoints to preserve.
        :param preserve_model:
            If ``True``, models in old checkpoints won't be deleted.
        """

    @abstractmethod
    def keep_best_n_checkpoints(self, n: int, *, preserve_model: bool = False) -> None:
        """Delete all but the best ``n`` checkpoints based on their score.

        :param n:
            The number of checkpoints to preserve.
        :param preserve_model:
            If ``True``, models in old checkpoints won't be deleted.
        """

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
    _lower_score_better: bool
    _checkpoint_step_nr: Optional[int]

    def __init__(
        self,
        checkpoint_dir: Path,
        gang: Gang,
        *,
        dp_gang: Optional[Gang] = None,
        tp_gang: Optional[Gang] = None,
        tensor_loader: Optional[TensorLoader] = None,
        tensor_dumper: Optional[TensorDumper] = None,
        lower_score_better: bool = False,
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
        :param lower_score_better:
            If ``True``, lower scores are considered better.
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

        self._lower_score_better = lower_score_better

        self._checkpoint_step_nr = None

    def save_model_metadata(
        self,
        *,
        base_asset: Optional[str] = None,
        family: Optional[str] = None,
        config: Optional[DataClass] = None,
        tokenizer_name: Optional[str] = None,
    ) -> None:
        """Set the model metadata.

        :param base_asset:
            The name of the asset that the model is based on.
        :param family:
            The family of the model.
        :param config:
            The configuration of the model.
        :param tokenizer_name:
            The name of the tokenizer that the model is trained with.
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

            if tokenizer_name is not None:
                metadata["tokenizer_ref"] = tokenizer_name

            try:
                self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            except OSError as ex:
                raise RuntimeError(
                    "The model metadata cannot be saved. See nested exception for details."
                ) from ex

            metadata_file = self._checkpoint_dir.joinpath("model.yaml")

            try:
                with metadata_file.open("w") as fp:
                    yaml.safe_dump(metadata, fp, sort_keys=False)
            except OSError as ex:
                raise RuntimeError(
                    "The model metadata cannot be saved. See nested exception for details."
                ) from ex

        self._root_gang.barrier()

    @override
    def begin_checkpoint(self, step_nr: int) -> None:
        if self._checkpoint_step_nr is not None:
            raise ValueError("`begin_checkpoint()` has already been called.")

        self.delete_checkpoint(step_nr, missing_ok=True)

        if self._root_gang.rank == 0:
            tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

            try:
                tmp_step_dir.mkdir(parents=True)
            except OSError as ex:
                raise RuntimeError(
                    f"The checkpoint directory of training step {step_nr} cannot be created. See nested exception for details."
                ) from ex

        self._root_gang.barrier()

        self._checkpoint_step_nr = step_nr

    @override
    def save_state(
        self,
        state: Mapping[str, Any],
        *,
        model_key: str = "model",
        replicated_keys: Optional[AbstractSet[str]] = None,
    ) -> None:
        step_nr = self._get_checkpoint_step_nr()

        def raise_error(cause: Exception) -> NoReturn:
            raise RuntimeError(
                f"The checkpoint of training step {step_nr} cannot be saved. See nested exception for details."
            ) from cause

        tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

        # Copy `state`. In case we fail, it should stay intact.
        rank_part = dict(state)

        rank_part["model_key"] = model_key

        def model_replicated() -> bool:
            if self._dp_gang.size == 1:
                return True

            if not replicated_keys:
                return False

            return model_key in replicated_keys or "*" in replicated_keys

        # Save the model into its own file if it is replicated.
        if model_replicated():
            state_dict = rank_part.pop(model_key, None)
            if state_dict is not None:
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
            skip_rank = len(rank_part) == 0
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

    @override
    def save_metadata(self, metadata: Mapping[str, Any]) -> None:
        step_nr = self._get_checkpoint_step_nr()

        if metadata is None:
            return

        if self._dp_gang.rank == 0:
            metadata_file = self._checkpoint_dir.joinpath(
                f"step_{step_nr}.tmp/metadata{self._shard_suffix}.pt"
            )

            try:
                self._tensor_dumper(metadata, metadata_file)
            except (RuntimeError, OSError, PickleError) as ex:
                raise RuntimeError(
                    f"The checkpoint metadata of training step {step_nr} cannot be saved. See nested exception for details."
                ) from ex

        self._root_gang.barrier()

    @override
    def save_score(self, score: Optional[float]) -> None:
        step_nr = self._get_checkpoint_step_nr()

        if self._root_gang.rank == 0:
            score_file = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp/score.txt")

            try:
                with score_file.open("w") as fp:
                    fp.write(f"{score}\n")
            except OSError as ex:
                raise RuntimeError(
                    f"The checkpoint score of training step {step_nr} cannot be saved. See nested exception for details.",
                    step_nr,
                ) from ex

        self._root_gang.barrier()

    @override
    def save_consolidated_fsdp_model(self, model: Module) -> None:
        step_nr = self._get_checkpoint_step_nr()

        log.info("Extracting consolidated model state.")

        with catch_warnings():
            warnings.simplefilter("ignore")  # Suppress noisy FSDP warnings.

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=True
                ),
            ):
                state_dict = model.state_dict()

        self._root_gang.barrier()

        log.info("Model state extracted. Saving to file on data parallel rank 0 (per shard).")  # fmt: skip

        if self._dp_gang.rank == 0:
            model_file = self._checkpoint_dir.joinpath(
                f"step_{step_nr}.tmp/model{self._shard_suffix}.pt"
            )

            try:
                self._tensor_dumper(
                    {"model": state_dict, "model_key": "model"}, model_file
                )
            except (RuntimeError, OSError, PickleError) as ex:
                raise RuntimeError(
                    f"The consolidated FSDP model of training step {step_nr} cannot be saved. See nested exception for details."
                ) from ex

        self._root_gang.barrier()

    @override
    def commit_checkpoint(self) -> None:
        step_nr = self._get_checkpoint_step_nr()

        if self._root_gang.rank == 0:
            tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

            step_dir = tmp_step_dir.with_suffix("")

            try:
                tmp_step_dir.replace(step_dir)
            except OSError as ex:
                raise RuntimeError(
                    f"The checkpoint of training step {step_nr} cannot be saved. See nested exception for details."
                ) from ex

        self._root_gang.barrier()

        self._checkpoint_step_nr = None

    def _get_checkpoint_step_nr(self) -> int:
        step_nr = self._checkpoint_step_nr
        if step_nr is None:
            raise ValueError("`begin_checkpoint()` must be called first.")

        return step_nr

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
    def delete_checkpoint(
        self, step_nr: int, *, missing_ok: bool = False, preserve_model: bool = False
    ) -> None:
        def raise_error(cause: Exception) -> NoReturn:
            raise RuntimeError(
                f"The checkpoint of training step {step_nr} cannot be deleted. See nested exception for details."
            ) from cause

        if self._root_gang.rank == 0:
            step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

            # Delete the temporary checkpoint directory if it exists.
            try:
                rmtree(step_dir.with_suffix(".tmp"))
            except OSError as ex:
                if not isinstance(ex, FileNotFoundError):
                    raise_error(ex)

            if not step_dir.exists():
                if not missing_ok:
                    raise RuntimeError(f"Training step {step_nr} has no checkpoint.")

                self._root_gang.barrier()

                return

            if preserve_model:
                # Delete all PyTorch tensor files except 'model.X.pt' files that
                # represent a (consolidated) model.
                for pt_file in step_dir.glob("*.pt"):
                    if pt_file.is_dir() or pt_file.stem.startswith("model"):
                        continue

                    try:
                        pt_file.unlink()
                    except OSError as ex:
                        if not isinstance(ex, FileNotFoundError):
                            raise_error(ex)
            else:
                try:
                    rmtree(step_dir)
                except OSError as ex:
                    if not missing_ok or not isinstance(ex, FileNotFoundError):
                        raise_error(ex)

        self._root_gang.barrier()

    @override
    def keep_last_n_checkpoints(self, n: int, *, preserve_model: bool = False) -> None:
        if n == 0:
            raise ValueError("`n` must be greater than zero.")

        step_numbers = self.get_step_numbers()
        if not step_numbers:
            return

        self._root_gang.barrier()

        for step_nr in step_numbers[:-n]:
            self.delete_checkpoint(step_nr, preserve_model=preserve_model)

    @override
    def keep_best_n_checkpoints(self, n: int, *, preserve_model: bool = False) -> None:
        if n == 0:
            raise ValueError("`n` must be greater than zero.")

        step_numbers = self.get_step_numbers()
        if not step_numbers:
            return

        last_step_nr = step_numbers[-1]

        scores = self._load_scores(step_numbers)
        if not scores:
            return

        self._root_gang.barrier()

        for _, step_nr in scores[:-n]:
            # Always preserve the last checkpoint.
            if step_nr != last_step_nr:
                self.delete_checkpoint(step_nr, preserve_model=preserve_model)

    def _load_scores(self, step_numbers: List[int]) -> List[Tuple[float, int]]:
        scores = []

        for step_nr in step_numbers:
            score_file = self._checkpoint_dir.joinpath(f"step_{step_nr}/score.txt")

            try:
                with score_file.open() as fp:
                    line = fp.readline()
            except OSError as ex:
                raise RuntimeError(
                    f"The score of training step {step_nr} cannot be loaded. See nested exception for details."
                ) from ex

            try:
                score = float(line)
            except ValueError as ex:
                raise RuntimeError(
                    f"The score of training step {step_nr} cannot be loaded. See nested exception for details."
                ) from ex

            scores.append((score, step_nr))

        if self._lower_score_better:
            scores.sort(key=lambda e: (-e[0], e[1]))
        else:
            scores.sort()

        return scores

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
