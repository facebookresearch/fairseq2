# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from os import scandir
from pathlib import Path
from shutil import Error
from typing import final

from typing_extensions import override

from fairseq2.gang import Gang, Gangs
from fairseq2.nn.data_parallel import load_with_sdp_gang
from fairseq2.typing import CPU
from fairseq2.utils.file import (
    FileMode,
    FileSystem,
    TensorDumper,
    TensorDumpError,
    TensorLoader,
    TensorLoadError,
)
from fairseq2.utils.state import Stateful


class CheckpointManager(ABC):
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
        metadata: Mapping[str, object] | None = None,
        score: float | None = None,
        lower_score_better: bool = False,
    ) -> None: ...

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
    def delete_checkpoint(self, step_nr: int, *, preserve_model: bool = False) -> None:
        """
        Deletes the checkpoint of the specified training step.

        :param step_nr: The number of the training step.
        :param preserve_model: If ``True``, model won't be deleted.
        """

    @abstractmethod
    def keep_last_n_checkpoints(self, n: int, *, preserve_model: bool = False) -> None:
        """
        Deletes all but the last ``n`` checkpoints.

        :param n: The number of checkpoints to preserve.
        :param preserve_model: If ``True``, models of old checkpoints won't be
            deleted.
        """

    @abstractmethod
    def keep_best_n_checkpoints(self, n: int, *, preserve_model: bool = False) -> None:
        """
        Deletes all but the best ``n`` checkpoints based on their score.

        :param n: The number of checkpoints to preserve.
        :param preserve_model: If ``True``, models of old checkpoints won't be
            deleted.
        """

    @abstractmethod
    def get_step_numbers(self) -> list[int]:
        """Returns the numbers of the training steps that have a checkpoint."""

    @abstractmethod
    def get_last_step_number(self) -> int | None: ...


@final
class FileCheckpointManager(CheckpointManager):
    """Saves and loads training checkpoints on a file system."""

    _checkpoint_dir: Path
    _gangs: Gangs
    _file_system: FileSystem
    _tensor_loader: TensorLoader
    _tensor_dumper: TensorDumper

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

    @override
    def save_checkpoint(
        self,
        step_nr: int,
        trainer: Stateful,
        model: Stateful,
        optimizer: Stateful,
        data_reader: Stateful,
        *,
        metadata: Mapping[str, object] | None = None,
        score: float | None = None,
        lower_score_better: bool = False,
    ) -> None:
        self._begin_checkpoint(step_nr)

        self._save_trainer_state(step_nr, trainer)

        self._save_model_state(step_nr, model)

        self._save_optimizer_state(step_nr, optimizer)

        self._save_data_reader_state(step_nr, data_reader)

        self._save_metadata(step_nr, metadata)

        self._save_score(step_nr, score, lower_score_better)

        self._copy_cc(step_nr)

        self._commit_checkpoint(step_nr)

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

            self._flush_nfs_lookup_cache()

        gangs.root.barrier()

        if gangs.root.rank != 0:
            self._flush_nfs_lookup_cache()

        self._checkpoint_step_nr = step_nr

    def _save_trainer_state(self, step_nr: int, trainer: Stateful) -> None:
        gangs = self._gangs

        trainer_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp/trainer")

        if gangs.root.rank == 0:
            try:
                self._file_system.make_directory(trainer_dir)
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The '{trainer_dir}' trainer directory of step {step_nr} cannot be created. See the nested exception for details."  # fmt: skip
                ) from ex

            self._flush_nfs_lookup_cache()

        gangs.root.barrier()

        if gangs.root.rank != 0:
            self._flush_nfs_lookup_cache()

        rank = self._rank_to_str(gangs.root)

        trainer_file = trainer_dir.joinpath(f"state_{rank}.pt")

        state_dict = trainer.state_dict()

        try:
            self._tensor_dumper.dump(state_dict, trainer_file)
        except TensorDumpError as ex:
            raise CheckpointSaveError(
                step_nr, f"The trainer state of step {step_nr} cannot be saved to the '{ex.path}' file. See the nested exception for details."  # fmt: skip
            ) from ex

        gangs.root.barrier()

    def _save_model_state(self, step_nr: int, model: Stateful) -> None:
        gangs = self._gangs

        if gangs.rdp.rank == 0:
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

                    self._flush_nfs_lookup_cache()

                gangs.sdp.barrier()

                if gangs.sdp.rank != 0:
                    self._flush_nfs_lookup_cache()

                model_file = model_path.joinpath(f"shard.{gangs.sdp.rank:02d}.pt")

            state_dict = model.state_dict()

            checkpoint = {"model": state_dict, "model_key": "model", "fs2": True}

            try:
                self._tensor_dumper.dump(checkpoint, model_file)
            except TensorDumpError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The model state of step {step_nr} cannot be saved to the '{ex.path}' file. See the nested exception for details."  # fmt: skip
                ) from ex

        gangs.root.barrier()

    def _save_optimizer_state(self, step_nr: int, optimizer: Stateful) -> None:
        gangs = self._gangs

        if gangs.rdp.rank == 0:
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

                    self._flush_nfs_lookup_cache()

                gangs.sdp.barrier()

                if gangs.sdp.rank != 0:
                    self._flush_nfs_lookup_cache()

                optimizer_file = optimizer_path.joinpath(
                    f"shard.{gangs.sdp.rank:02d}.pt"
                )

            state_dict = optimizer.state_dict()

            try:
                self._tensor_dumper.dump(state_dict, optimizer_file)
            except TensorDumpError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The optimizer state of step {step_nr} cannot be saved to the '{ex.path}' file. See the nested exception for details."  # fmt: skip
                ) from ex

        gangs.root.barrier()

    def _save_data_reader_state(self, step_nr: int, data_reader: Stateful) -> None:
        gangs = self._gangs

        if gangs.tp.rank == 0:
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

                self._flush_nfs_lookup_cache()

            gangs.dp.barrier()

            if gangs.dp.rank != 0:
                self._flush_nfs_lookup_cache()

            rank = self._rank_to_str(gangs.dp)

            data_reader_file = data_reader_dir.joinpath(f"state_{rank}.pt")

            state_dict = data_reader.state_dict()

            try:
                self._tensor_dumper.dump(state_dict, data_reader_file)
            except TensorDumpError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The data reader state of step {step_nr} cannot be saved to the '{ex.path}' file. See the nested exception for details."  # fmt: skip
                ) from ex

        gangs.root.barrier()

    def _save_metadata(
        self, step_nr: int, metadata: Mapping[str, object] | None
    ) -> None:
        gangs = self._gangs

        if gangs.root.rank == 0 and metadata is not None:
            metadata_file = self._checkpoint_dir.joinpath(
                f"step_{step_nr}.tmp/metadata.pt"
            )

            try:
                self._tensor_dumper.dump(metadata, metadata_file)
            except TensorDumpError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The metadata of step {step_nr} cannot be saved to the '{ex.path}' file. See the nested exception for details."  # fmt: skip
                ) from ex

        gangs.root.barrier()

    def _save_score(
        self, step_nr: int, score: float | None, lower_better: bool
    ) -> None:
        gangs = self._gangs

        if gangs.root.rank == 0 and score is not None:
            score_file = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp/score.txt")

            try:
                fp = self._file_system.open_text(score_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The score of step {step_nr} cannot be saved to the '{score_file}' file. See the nested exception for details."  # fmt: skip
                ) from ex

            try:
                fp.write(f"{'-' if lower_better else ''}{score}\n")
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The score of step {step_nr} cannot be saved to the '{score_file}' file. See the nested exception for details."  # fmt: skip
                ) from ex
            finally:
                fp.close()

        self._gangs.root.barrier()

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

        gangs.root.barrier()

    def _commit_checkpoint(self, step_nr: int) -> None:
        gangs = self._gangs

        if gangs.root.rank == 0:
            tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

            step_dir = tmp_step_dir.with_suffix("")

            try:
                self._file_system.move(tmp_step_dir, step_dir)
            except OSError as ex:
                raise CheckpointSaveError(
                    step_nr, f"The temporary '{tmp_step_dir}' checkpoint directory of step {step_nr} cannot be committed. See the nested exception for details."  # fmt: skip
                ) from ex

        gangs.root.barrier()

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

        with load_with_sdp_gang(gangs):  # Required for ShardedTensor.
            file = self._checkpoint_dir.joinpath(f"step_{step_nr}/{pathname}")

            try:
                state_dict = self._tensor_loader.load(
                    file, map_location=CPU, restrict=False
                )
            except FileNotFoundError:
                raise CheckpointNotFoundError(step_nr) from None

        gangs.root.barrier()

        return state_dict

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
    def delete_checkpoint(self, step_nr: int, *, preserve_model: bool = False) -> None:
        gangs = self._gangs

        if gangs.root.rank == 0:
            step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

            tmp_step_dir = step_dir.with_suffix(".tmp")

            try:
                self._file_system.remove_directory(tmp_step_dir)
            except FileNotFoundError:
                pass
            except OSError as ex:
                raise CheckpointDeleteError(
                    step_nr, f"The temporary '{tmp_step_dir}' checkpoint directory of step {step_nr} cannot be deleted. See the nested exception for details."  # fmt: skip
                ) from ex

            try:
                step_dir_exists = self._file_system.exists(step_dir)
            except OSError as ex:
                raise CheckpointDeleteError(
                    step_nr, f"The '{step_dir}' checkpoint directory of step {step_nr} cannot be accessed. See the nested exception for details."  # fmt: skip
                ) from ex

            if not step_dir_exists:
                gangs.root.barrier()

                return

            if preserve_model:
                for pathname in ["trainer", "optimizer.pt", "data_reader", "score.txt"]:
                    path = step_dir.joinpath(pathname)

                    try:
                        if self._file_system.is_dir(path):
                            self._file_system.remove_directory(path)
                        else:
                            try:
                                self._file_system.remove(path)
                            except FileNotFoundError:
                                pass
                    except OSError as ex:
                        raise CheckpointDeleteError(
                            step_nr, f"The '{path}' path of step {step_nr} cannot be deleted. See the nested exception for details."  # fmt: skip
                        ) from ex
            else:
                try:
                    self._file_system.remove_directory(step_dir)
                except OSError as ex:
                    raise CheckpointDeleteError(
                        step_nr, f"The '{step_dir}' checkpoint directory of step {step_nr} cannot be deleted. See the nested exception for details."  # fmt: skip
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
        if n < 1:
            raise ValueError("`n` must be greater than or equal to 1.")

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
                    f"The score of step {step_nr} cannot be loaded from the '{score_file}' file. See the nested exception for details."
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
                    f"The score of step {step_nr} cannot be parsed as a floating-point number."
                ) from None

            scores.append((score, step_nr))

        scores.sort()

        return scores

    @override
    def get_step_numbers(self) -> list[int]:
        step_numbers = []

        try:
            for step_dir in self._file_system.glob(self._checkpoint_dir, "step_*"):
                if not self._file_system.is_dir(step_dir):
                    continue

                try:
                    step_nr = int(step_dir.name[5:])
                except ValueError:
                    continue

                step_numbers.append(step_nr)
        except OSError as ex:
            raise CheckpointError(
                f"The '{self._checkpoint_dir}' checkpoint directory cannot be traversed. See the nested exception for details."
            ) from ex

        step_numbers.sort()

        return step_numbers

    @override
    def get_last_step_number(self) -> int | None:
        step_numbers = self.get_step_numbers()
        if step_numbers:
            return step_numbers[-1]

        return None

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
