# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from pathlib import Path
from pickle import PickleError
from typing import Any, Dict, Optional, Tuple, cast, final

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
    def save_checkpoint(self, step_nr: int, state: Dict[str, Any]) -> None:
        """Save a checkpoint.

        :param step_nr:
            The training step number of the checkpoint.
        :param state:
            The checkpoint state.
        """

    @abstractmethod
    def load_checkpoint(self, step_nr: int) -> Dict[str, Any]:
        """Load the checkpoint of the specified training step.

        :param step_nr:
            The training step number.
        """

    @abstractmethod
    def load_last_checkpoint(self) -> Tuple[int, Dict[str, Any]]:
        """Load the last checkpoint.

        :returns:
            - The training step number of the checkpoint.
            - The checkpoint state.
        """

    @abstractmethod
    def get_last_step_nr(self) -> Optional[int]:
        """Return the training step number of the last checkpoint."""

    @abstractmethod
    def has_last_checkpoint(self) -> bool:
        """Return ``True`` if there is a checkpoint marked as last."""

    @abstractmethod
    def save_consolidated_fsdp_model(self, step_nr: int, model: Module) -> None:
        """Save ``model`` with a consolidated ``state_dict``.

        This method consolidates the state of the model from all ranks so that
        it can be loaded on a single device without FSDP.
        """

    @abstractmethod
    def load_consolidated_model(
        self, step_nr: int, model: Module, device: Optional[Device] = None
    ) -> None:
        """Load the consolidated model of the specified training step.

        :param step_nr:
            The training step number.
        :param model:
            The model to load.
        :param device:
            The device on which to load the model if it is on a meta device.
        """

    @abstractmethod
    def load_last_consolidated_model(
        self, model: Module, device: Optional[Device] = None
    ) -> int:
        """Load the last consolidated model.

        :param model:
            The model to load.
        :param device:
            The device on which to load the model if it is on a meta device.

        :returns:
            The training step number of the consolidated model.
        """


@final
class FileCheckpointManager(CheckpointManager):
    """Saves and loads training checkpoints on a file system."""

    checkpoint_dir: Path
    gang: Gang

    def __init__(self, checkpoint_dir: Path, gang: Gang) -> None:
        """
        :param checkpoint_dir:
            The root directory under which to store the checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        self.gang = gang

    @finaloverride
    def save_checkpoint(self, step_nr: int, state: Dict[str, Any]) -> None:
        checkpoint = {"step_nr": step_nr, "data": state}

        step_dir = self.checkpoint_dir.joinpath(f"step_{step_nr}")

        if self.gang.rank == 0:
            try:
                step_dir.mkdir(parents=True, exist_ok=True)
            except IOError as ex:
                raise RuntimeError(
                    f"The checkpoint directory for step {step_nr} cannot be created. See nested exception for details."
                ) from ex

        self.gang.barrier()

        checkpoint_file = step_dir.joinpath(f"rank_{self.gang.rank}.pt")

        try:
            torch.save(checkpoint, checkpoint_file)
        except (IOError, PickleError) as ex:
            raise RuntimeError(
                f"The checkpoint of step {step_nr} cannot be saved. See nested exception for details."
            ) from ex

        self.gang.barrier()

        if self.gang.rank == 0:
            last_symlink = self.checkpoint_dir.joinpath("last")
            temp_symlink = self.checkpoint_dir.joinpath("last.temp")

            try:
                temp_symlink.unlink(missing_ok=True)

                temp_symlink.symlink_to(step_dir.name, target_is_directory=True)

                temp_symlink.replace(last_symlink)
            except IOError as ex:
                raise RuntimeError(
                    f"The checkpoint of step {step_nr} cannot be marked as last. See nested exception for details."
                ) from ex

        self.gang.barrier()

    @finaloverride
    def load_checkpoint(self, step_nr: int) -> Dict[str, Any]:
        step_dir = self.checkpoint_dir.joinpath(f"step_{step_nr}")
        if not step_dir.exists():
            raise CheckpointNotFoundError(f"Step {step_nr} has no checkpoint.")

        checkpoint_file = step_dir.joinpath(f"rank_{self.gang.rank}.pt")

        try:
            checkpoint = load_checkpoint(checkpoint_file, map_location=CPU)
        except (IOError, PickleError) as ex:
            raise RuntimeError(
                f"The checkpoint of step {step_nr} cannot be loaded. See nested exception for details."
            ) from ex

        try:
            saved_step_nr = checkpoint["step_nr"]
        except KeyError:
            raise RuntimeError(
                f"The checkpoint of step {step_nr} does not contain a step number."
            )

        if not isinstance(saved_step_nr, int):
            raise RuntimeError(
                f"The checkpoint of step {step_nr} does not contain a valid step number."
            )

        if saved_step_nr != step_nr:
            raise RuntimeError(
                f"The saved step number ({saved_step_nr}) in the checkpoint does not match the step number ({step_nr}) in the directory name."
            )

        try:
            state = cast(Dict[str, Any], checkpoint["data"])
        except KeyError:
            raise RuntimeError(
                f"The checkpoint of step {step_nr} does not contain any data."
            )

        self.gang.barrier()

        return state

    @finaloverride
    def load_last_checkpoint(self) -> Tuple[int, Dict[str, Any]]:
        last_step_nr = self.get_last_step_nr()
        if last_step_nr is None:
            raise CheckpointNotFoundError("There is no checkpoint marked as last.")

        checkpoint = self.load_checkpoint(last_step_nr)

        return last_step_nr, checkpoint

    @finaloverride
    def get_last_step_nr(self) -> Optional[int]:
        step_dir = self.checkpoint_dir.joinpath("last").resolve()
        if not step_dir.exists():
            return None

        dirname_parts = step_dir.name.split("_", 1)
        if len(dirname_parts) == 2:
            try:
                return int(dirname_parts[1])
            except ValueError:
                pass

        raise RuntimeError(
            f"The name of the checkpoint directories must be in format 'step_<nr>', but the last checkpoint directory is named {step_dir.name}."
        )

    @finaloverride
    def has_last_checkpoint(self) -> bool:
        return self.checkpoint_dir.joinpath("last").exists()

    @finaloverride
    def save_consolidated_fsdp_model(self, step_nr: int, model: Module) -> None:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state = model.state_dict()

        checkpoint = {"step_nr": step_nr, "model": state}

        if self.gang.rank == 0:
            step_dir = self.checkpoint_dir.joinpath(f"step_{step_nr}")

            try:
                step_dir.mkdir(parents=True, exist_ok=True)
            except IOError as ex:
                raise RuntimeError(
                    f"The checkpoint directory of step {step_nr} cannot be created. See nested exception for details."
                ) from ex

            checkpoint_file = step_dir.joinpath("model.pt")

            try:
                torch.save(checkpoint, checkpoint_file)
            except (IOError, PickleError) as ex:
                raise RuntimeError(
                    f"The consolidated model checkpoint of step {step_nr} cannot be saved. See nested exception for details."
                ) from ex

        self.gang.barrier()

    @finaloverride
    def load_consolidated_model(
        self, step_nr: int, model: Module, device: Optional[Device] = None
    ) -> None:
        if self.gang.rank == 0:
            model_file = self.checkpoint_dir.joinpath(f"step_{step_nr}/model.pt")
            if not model_file.exists():
                raise CheckpointNotFoundError(
                    f"Step {step_nr} has no consolidated model checkpoint."
                )

            try:
                checkpoint = load_checkpoint(model_file, map_location=CPU)
            except (IOError, PickleError) as ex:
                raise RuntimeError(
                    f"The consolidated model checkpoint of step {step_nr} cannot be loaded. See nested exception for details."
                ) from ex

            model_device = infer_device(model)

            if model_device == META:
                # Move the model to the actual device without initializing. Its
                # state will be overwritten by the checkpoint anyways.
                to_empty(model, device=device or CPU)

            # Load the model.
            try:
                state_dict = checkpoint["model"]
            except KeyError:
                raise RuntimeError(
                    f"The consolidated model checkpoint of step {step_nr} does not contain a 'model' entry."
                )

            try:
                model.load_state_dict(state_dict)
            except (KeyError, ValueError) as ex:
                raise RuntimeError(
                    f"The consolidated model checkpoint of step {step_nr} cannot be loaded. See nested exception for details."
                ) from ex

            if model_device == META:
                # Non-persistent buffers are not included in the checkpoint, so
                # we have to explicitly initialize them.
                reset_non_persistent_buffers(model)

        self.gang.barrier()

    @finaloverride
    def load_last_consolidated_model(
        self, model: Module, device: Optional[Device] = None
    ) -> int:
        last_step_nr = self.get_last_step_nr()
        if last_step_nr is None:
            raise CheckpointNotFoundError("There is no checkpoint marked as last.")

        self.load_consolidated_model(last_step_nr, model, device)

        return last_step_nr


class CheckpointNotFoundError(RuntimeError):
    """Raised when a checkpoint is not found."""
