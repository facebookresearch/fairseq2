# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Set
from contextlib import AbstractContextManager, nullcontext
from os import scandir
from pathlib import Path
from shutil import Error
from typing import final

from torch.distributed._shard import load_with_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.nn import Module
from typing_extensions import override

from fairseq2.error import InvalidOperationError, NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.typing import CPU
from fairseq2.utils.file import (
    FileMode,
    FileSystem,
    TensorDumper,
    TensorDumpError,
    TensorLoader,
    TensorLoadError,
)


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
        self, state: Mapping[str, object], *, replicated_keys: Set[str] | None = None
    ) -> None:
        """Save the training state.

        :param state:
            The state to save.
        :param replicated_keys:
            The keys in ``state`` whose values are replicated across all
            processes in the gang.
        """

    @abstractmethod
    def save_model(self, model: Module) -> None:
        """Save ``model``."""

    @abstractmethod
    def save_consolidated_fsdp_model(self, model: FSDP) -> None:
        """Save ``model``."""

    @abstractmethod
    def save_metadata(self, metadata: Mapping[str, object]) -> None:
        """Save ``metadata`` associated with the checkpoint.

        :param metadata:
            The metadata to save. Must be pickeable.
        """

    @abstractmethod
    def save_score(self, score: float | None, lower_better: bool = False) -> None:
        """Save the score of the checkpoint."""

    @abstractmethod
    def commit_checkpoint(self) -> None:
        """Commit the checkpoint after which it will be considered saved."""

    @abstractmethod
    def load_checkpoint(
        self, step_nr: int, *, load_model: bool = True
    ) -> dict[str, object]:
        """Load the checkpoint of the specified training step."""

    @abstractmethod
    def load_last_checkpoint(
        self, *, load_model: bool = True
    ) -> tuple[int, dict[str, object]]:
        """Load the last checkpoint in the training.

        :returns:
            - The number of the training step.
            - The checkpoint.
        """

    @abstractmethod
    def get_model_path(self, step_nr: int) -> Path: ...

    @abstractmethod
    def get_last_model_path(self) -> tuple[int, Path]: ...

    @abstractmethod
    def load_metadata(self, step_nr: int) -> dict[str, object] | None:
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
    def has_checkpoint(self, step_nr: int | None = None) -> bool:
        """Return ``True`` if the manager holds a checkpoint.

        :param step_nr:
            The number of the training step. If ``None``, returns ``True`` if
            the manager holds at least one checkpoint.
        """

    @abstractmethod
    def get_step_numbers(self) -> list[int]:
        """Return the numbers of the training steps that have a checkpoint."""


@final
class FileCheckpointManager(CheckpointManager):
    """Saves and loads training checkpoints on a file system."""

    _checkpoint_dir: Path
    _gangs: Gangs
    _file_system: FileSystem
    _tensor_loader: TensorLoader
    _tensor_dumper: TensorDumper
    _shard_suffix: str
    _checkpoint_step_nr: int | None

    def __init__(
        self,
        checkpoint_dir: Path,
        gangs: Gangs,
        file_system: FileSystem,
        tensor_loader: TensorLoader,
        tensor_dumper: TensorDumper,
    ) -> None:
        try:
            self._checkpoint_dir = file_system.resolve(checkpoint_dir)
        except OSError as ex:
            raise CheckpointError(
                f"'{checkpoint_dir}' cannot be accessed. See the nested exception for details."
            ) from ex

        self._gangs = gangs

        self._file_system = file_system

        self._tensor_loader = tensor_loader
        self._tensor_dumper = tensor_dumper

        if gangs.tp.size > 1:
            self._shard_suffix = f".{gangs.tp.rank}"
        else:
            self._shard_suffix = ""

        self._checkpoint_step_nr = None

    @override
    def begin_checkpoint(self, step_nr: int) -> None:
        if self._checkpoint_step_nr is not None:
            raise InvalidOperationError("`begin_checkpoint()` has already been called.")

        try:
            self.delete_checkpoint(step_nr, missing_ok=True)
        except CheckpointError as ex:
            raise CheckpointSaveError(
                step_nr, f"The previous checkpoint of training step {step_nr} cannot be deleted. See the nested exception for details."  # fmt: skip
            ) from ex

        tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

        if self._gangs.root.rank == 0:
            try:
                self._file_system.make_directory(tmp_step_dir)
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The temporary '{tmp_step_dir}' checkpoint directory of training step {step_nr} cannot be created. See the nested exception for details."  # fmt: skip
                ) from ex

            self._flush_nfs_lookup_cache()

        self._gangs.root.barrier()

        # Ensure that `tmp_step_dir` is visible to all processes.
        if self._gangs.root.rank != 0:
            self._flush_nfs_lookup_cache()

        self._checkpoint_step_nr = step_nr

    @override
    def save_state(
        self, state: Mapping[str, object], *, replicated_keys: Set[str] | None = None
    ) -> None:
        gangs = self._gangs

        step_nr = self._get_checkpoint_step_nr()

        tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

        suffix = self._shard_suffix

        # Copy `state`. In case we fail, it should stay intact.
        rank_part = dict(state)

        def model_replicated() -> bool:
            if gangs.dp.size == 1:
                return True

            if not replicated_keys:
                return False

            return "model" in replicated_keys or "*" in replicated_keys

        # Save the model into its own file if it is replicated.
        if model_replicated():
            model_state = rank_part.pop("model", None)
            if model_state is not None:
                if gangs.dp.rank == 0:
                    model_file = tmp_step_dir.joinpath(f"model{suffix}.pt")

                    model_checkpoint = {
                        "model": model_state, "model_key": "model", "fs2": True  # fmt: skip
                    }

                    try:
                        self._tensor_dumper.dump(model_checkpoint, model_file)
                    except TensorDumpError as ex:
                        raise CheckpointSaveError(
                            step_nr, f"The model of training step {step_nr} cannot be saved to the '{model_file}' file. See the nested exception for details."  # fmt: skip
                        ) from ex

                gangs.root.barrier()

        # Save the replicated state.
        if replicated_keys:
            if gangs.dp.rank == 0:
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
                    replicated_file = tmp_step_dir.joinpath(f"replicated{suffix}.pt")

                    try:
                        self._tensor_dumper.dump(replicated_part, replicated_file)
                    except TensorDumpError as ex:
                        raise CheckpointSaveError(
                            step_nr, f"The replicated checkpoint state of training step {step_nr} cannot be saved to the '{replicated_file}' file. See the nested exception for details."  # fmt: skip
                        ) from ex
            else:
                if "*" in replicated_keys:
                    rank_part.clear()
                else:
                    for key in replicated_keys:
                        try:
                            del rank_part[key]
                        except KeyError:
                            pass

            gangs.root.barrier()

            # Check if anything is left to save for the rank.
            skip_rank = len(rank_part) == 0
        else:
            skip_rank = False

        # Save the per-rank state.
        if not skip_rank:
            rank_file = tmp_step_dir.joinpath(f"rank_{gangs.dp.rank}{suffix}.pt")

            try:
                self._tensor_dumper.dump(rank_part, rank_file)
            except TensorDumpError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The checkpoint state of training step {step_nr} cannot be saved to the '{rank_file}' file. See the nested exception for details."  # fmt: skip
                ) from ex

            gangs.root.barrier()

        # Copy carbon-copy files to the checkpoint directory.
        if gangs.root.rank == 0:
            cc_dir = self._checkpoint_dir.joinpath("cc")

            try:
                self._file_system.copy_directory(cc_dir, tmp_step_dir)
            except FileNotFoundError:
                pass
            except (OSError, Error) as ex:
                raise CheckpointSaveError(
                    step_nr, f"The checkpoint carbon copy directory cannot be copied to the '{tmp_step_dir}' directory. See the nested exception for details."  # fmt: skip
                ) from ex

        gangs.root.barrier()

    @override
    def save_model(self, model: Module) -> None:
        gangs = self._gangs

        if isinstance(model, FSDP):
            log.info("Extracting the consolidated FSDP model state on rank 0.")

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=r".*FSDP\.state_dict_type\(\) and FSDP\.set_state_dict_type\(\) are being deprecated.*"  # fmt: skip
                )

                state_dict_config = FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=True
                )

                with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT, state_dict_config
                ):
                    model_state = model.state_dict()

            log.info("Model state extracted.")
        else:
            if gangs.dp.rank == 0:
                model_state = model.state_dict()
            else:
                model_state = {}

        gangs.root.barrier()

        step_nr = self._get_checkpoint_step_nr()

        if gangs.dp.rank == 0:
            model_file = self._checkpoint_dir.joinpath(
                f"step_{step_nr}.tmp/model{self._shard_suffix}.pt"
            )

            model_checkpoint = {"model": model_state, "model_key": "model", "fs2": True}

            try:
                self._tensor_dumper.dump(model_checkpoint, model_file)
            except TensorDumpError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The model of training step {step_nr} cannot be saved to the '{model_file}' file. See the nested exception for details."  # fmt: skip
                ) from ex

        gangs.root.barrier()

    @override
    def save_consolidated_fsdp_model(self, model: FSDP) -> None:
        self.save_model(model)

    @override
    def save_metadata(self, metadata: Mapping[str, object]) -> None:
        step_nr = self._get_checkpoint_step_nr()

        if metadata is None:
            return

        if self._gangs.dp.rank == 0:
            metadata_file = self._checkpoint_dir.joinpath(
                f"step_{step_nr}.tmp/metadata{self._shard_suffix}.pt"
            )

            try:
                self._tensor_dumper.dump(metadata, metadata_file)
            except TensorDumpError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The checkpoint metadata of training step {step_nr} cannot be saved to the '{metadata_file}' file. See the nested exception for details."  # fmt: skip
                ) from ex

        self._gangs.root.barrier()

    @override
    def save_score(self, score: float | None, lower_better: bool = False) -> None:
        step_nr = self._get_checkpoint_step_nr()

        if self._gangs.root.rank == 0 and score is not None:
            score_file = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp/score.txt")

            def save_error() -> CheckpointError:
                return CheckpointSaveError(
                    step_nr, f"The checkpoint score of training step {step_nr} cannot be saved to the '{score_file}' file. See the nested exception for details."  # fmt: skip
                )

            try:
                fp = self._file_system.open_text(score_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise save_error() from ex

            direction = "-" if lower_better else ""

            try:
                fp.write(f"{direction}{score}\n")
            except OSError as ex:
                raise save_error() from ex
            finally:
                fp.close()

        self._gangs.root.barrier()

    @override
    def commit_checkpoint(self) -> None:
        step_nr = self._get_checkpoint_step_nr()

        if self._gangs.root.rank == 0:
            tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

            step_dir = tmp_step_dir.with_suffix("")

            try:
                self._file_system.move(tmp_step_dir, step_dir)
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The temporary '{tmp_step_dir}' checkpoint directory of training step {step_nr} cannot be committed. See the nested exception for details."  # fmt: skip
                ) from ex

        self._gangs.root.barrier()

        self._checkpoint_step_nr = None

    def _get_checkpoint_step_nr(self) -> int:
        step_nr = self._checkpoint_step_nr
        if step_nr is None:
            raise InvalidOperationError("`begin_checkpoint()` must be called first.")

        return step_nr

    @override
    def load_checkpoint(
        self, step_nr: int, *, load_model: bool = True
    ) -> dict[str, object]:
        gangs = self._gangs

        def maybe_with_dp_process_group() -> AbstractContextManager[None]:
            try:
                pg = gangs.dp.as_process_group()
            except NotSupportedError:
                return nullcontext()

            return load_with_process_group(pg)

        step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

        def maybe_load_part(filename: str) -> dict[str, object]:
            with maybe_with_dp_process_group():  # Required for `ShardedTensor`.
                file = step_dir.joinpath(filename)

                try:
                    part = self._tensor_loader.load(file, map_location=CPU)
                except FileNotFoundError:
                    part = {}
                except TensorLoadError as ex:
                    raise CheckpointLoadError(
                        step_nr, f"The '{file}' checkpoint file of training step {step_nr} cannot be loaded. See the nested exception for details."  # fmt: skip
                    ) from ex

                gangs.root.barrier()

                return part

        checkpoint = {}

        suffix = self._shard_suffix

        for file in [f"rank_{gangs.dp.rank}{suffix}.pt", f"replicated{suffix}.pt"]:
            part = maybe_load_part(file)

            # Consolidate the checkpoint parts.
            checkpoint.update(part)

        if load_model:
            # If we don't have the model state in the checkpoint so far, it
            # means it was replicated (e.g. DDP).
            if "model" not in checkpoint:
                part = maybe_load_part(f"model{suffix}.pt")

                checkpoint.update(part)
        else:
            try:
                del checkpoint["model"]
            except KeyError:
                pass

        if not checkpoint:
            raise CheckpointNotFoundError(step_nr)

        return checkpoint

    @override
    def load_last_checkpoint(
        self, *, load_model: bool = True
    ) -> tuple[int, dict[str, object]]:
        step_numbers = self.get_step_numbers()
        if not step_numbers:
            raise CheckpointNotFoundError()

        last_step_nr = step_numbers[-1]

        checkpoint = self.load_checkpoint(last_step_nr, load_model=load_model)

        return last_step_nr, checkpoint

    @override
    def get_model_path(self, step_nr: int) -> Path:
        model_file = self._checkpoint_dir.joinpath(
            f"step_{step_nr}/model{self._shard_suffix}.pt"
        )

        try:
            model_file_exists = self._file_system.exists(model_file)
        except OSError as ex:
            raise CheckpointError(
                f"The '{model_file}' path cannot be accessed. See the nested exception for details."
            ) from ex

        if not model_file_exists:
            raise CheckpointNotFoundError(step_nr)

        return model_file

    @override
    def get_last_model_path(self) -> tuple[int, Path]:
        step_numbers = self.get_step_numbers()
        if not step_numbers:
            raise CheckpointNotFoundError()

        last_step_nr = step_numbers[-1]

        path = self.get_model_path(last_step_nr)

        return last_step_nr, path

    @override
    def load_metadata(self, step_nr: int) -> dict[str, object] | None:
        metadata_file = self._checkpoint_dir.joinpath(
            f"step_{step_nr}/metadata{self._shard_suffix}.pt"
        )

        try:
            metadata = self._tensor_loader.load(metadata_file, map_location=CPU)
        except FileNotFoundError:
            metadata = None
        except TensorLoadError as ex:
            raise CheckpointLoadError(
                step_nr, f"The checkpoint metadata of training step {step_nr} cannot be loaded from the '{metadata_file}' file. See the nested exception for details."  # fmt: skip
            ) from ex

        self._gangs.root.barrier()

        return metadata

    @override
    def delete_checkpoint(
        self, step_nr: int, *, missing_ok: bool = False, preserve_model: bool = False
    ) -> None:
        gangs = self._gangs

        if gangs.root.rank == 0:
            step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

            # Delete the temporary checkpoint directory if it exists.
            tmp_step_dir = step_dir.with_suffix(".tmp")

            def delete_error() -> CheckpointError:
                return CheckpointDeleteError(
                    step_nr, f"The temporary '{tmp_step_dir}' checkpoint directory of training step {step_nr} cannot be deleted. See the nested exception for details."  # fmt: skip
                )

            try:
                self._file_system.remove_directory(tmp_step_dir)
            except FileNotFoundError:
                pass
            except OSError as ex:
                raise delete_error() from ex

            try:
                step_dir_exists = self._file_system.exists(step_dir)
            except OSError as ex:
                raise delete_error() from ex

            if not step_dir_exists:
                if not missing_ok:
                    raise CheckpointDeleteError(
                        step_nr, f"The '{step_dir}' checkpoint directory of training step {step_nr} is not found."  # fmt: skip
                    )

                gangs.root.barrier()

                return

            if preserve_model:

                def iter_torch_files() -> Iterable[Path]:
                    try:
                        for pt_file in self._file_system.glob(step_dir, "*.pt"):
                            if self._file_system.is_dir(pt_file):
                                continue

                            yield pt_file
                    except OSError as ex:
                        raise CheckpointDeleteError(
                            step_nr, f"The '{step_dir}' checkpoint directory of training step {step_nr} cannot be traversed. See the nested exception for details."  # fmt: skip
                        ) from ex

                # Delete all PyTorch tensor files except 'model.X.pt' files that
                # represent a (consolidated) model.
                for pt_file in iter_torch_files():
                    if pt_file.stem.startswith("model"):
                        continue

                    try:
                        self._file_system.remove(pt_file)
                    except FileNotFoundError:
                        pass
                    except OSError as ex:
                        raise CheckpointDeleteError(
                            step_nr, f"The '{pt_file}' checkpoint file of training step {step_nr} cannot be deleted. See the nested exception for details."  # fmt: skip
                        ) from ex
            else:
                try:
                    self._file_system.remove_directory(step_dir)
                except FileNotFoundError:
                    pass
                except OSError as ex:
                    if not missing_ok:
                        raise CheckpointDeleteError(
                            step_nr, f"The '{step_dir}' checkpoint directory of training step {step_nr} cannot be deleted. See the nested exception for details."  # fmt: skip
                        ) from ex

        gangs.root.barrier()

    @override
    def keep_last_n_checkpoints(self, n: int, *, preserve_model: bool = False) -> None:
        if n == 0:
            raise ValueError("`n` must be greater than zero.")

        step_numbers = self.get_step_numbers()
        if not step_numbers:
            return

        self._gangs.root.barrier()

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

        self._gangs.root.barrier()

        for _, step_nr in scores[:-n]:
            # Always preserve the last checkpoint.
            if step_nr != last_step_nr:
                self.delete_checkpoint(step_nr, preserve_model=preserve_model)

    def _load_scores(self, step_numbers: list[int]) -> list[tuple[float, int]]:
        scores = []

        for step_nr in step_numbers:
            score_file = self._checkpoint_dir.joinpath(f"step_{step_nr}/score.txt")

            def load_error() -> CheckpointError:
                return CheckpointError(
                    f"The score of training step {step_nr} cannot be loaded from the '{score_file}' file. See the nested exception for details."
                )

            try:
                fp = self._file_system.open_text(score_file)
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
                    f"The score of training step {step_nr} cannot be parsed as a floating-point number."
                ) from None

            scores.append((score, step_nr))

        scores.sort()

        return scores

    @override
    def has_checkpoint(self, step_nr: int | None = None) -> bool:
        it = self._iter_step_numbers()

        if step_nr is None:
            return next(it, None) is not None

        return step_nr in it

    @override
    def get_step_numbers(self) -> list[int]:
        step_numbers = list(self._iter_step_numbers())

        step_numbers.sort()

        return step_numbers

    def _iter_step_numbers(self) -> Iterator[int]:
        try:
            for step_dir in self._file_system.glob(self._checkpoint_dir, "step_*"):
                if not self._file_system.is_dir(step_dir):
                    continue

                try:
                    step_nr = int(step_dir.name[5:])
                except ValueError:
                    continue

                yield step_nr
        except OSError as ex:
            raise CheckpointError(
                f"The base '{self._checkpoint_dir}' checkpoint directory cannot be traversed. See the nested exception for details."
            ) from ex

    def _flush_nfs_lookup_cache(self) -> None:
        if not self._file_system.is_local:
            return

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


class CheckpointNotFoundError(Exception):
    step_nr: int | None

    def __init__(self, step_nr: int | None = None) -> None:
        if step_nr is None:
            message = "No checkpoint found."
        else:
            message = f"No checkpoint found for training step {step_nr}."

        super().__init__(message)

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
