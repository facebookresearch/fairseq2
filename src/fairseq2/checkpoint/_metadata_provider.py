# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, final

from fairseq2.assets import (
    AssetMetadataFileLoader,
    AssetMetadataLoadError,
    AssetMetadataProvider,
    AssetMetadataSaveError,
    CachedAssetMetadataProvider,
)
from fairseq2.file_system import FileSystem
from fairseq2.gang import GangError, Gangs
from fairseq2.utils.structured import unstructure
from fairseq2.utils.yaml import YamlDumper


class CheckpointMetadataSaver(ABC):
    @abstractmethod
    def save(self, model_family: str, model_config: object) -> None: ...


@final
class FileCheckpointMetadataSaver(CheckpointMetadataSaver):
    _checkpoint_dir: Path
    _gangs: Gangs
    _file_system: FileSystem
    _yaml_dumper: YamlDumper

    def __init__(
        self,
        checkpoint_dir: Path,
        gangs: Gangs,
        file_system: FileSystem,
        yaml_dumper: YamlDumper,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._gangs = gangs
        self._file_system = file_system
        self._yaml_dumper = yaml_dumper

    def save(self, model_family: str, model_config: object) -> None:
        if self._gangs.root.rank == 0:
            self._save_asset_card(model_family, model_config)

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise AssetMetadataSaveError(
                "The collective barrier after the checkpoint metadata save operation has failed. See the nested exception for details."
            ) from ex

    def _save_asset_card(self, model_family: str, model_config: object) -> None:
        unstructured_model_config = unstructure(model_config)

        metadata: dict[str, object] = {
            "name": "checkpoint",
            "model_family": model_family,
            "model_config": {
                "_set_": unstructured_model_config,
            },
        }

        metadata_file = self._checkpoint_dir.joinpath("model.yaml")

        def save_error() -> AssetMetadataSaveError:
            return AssetMetadataSaveError(
                f"The checkpoint metadata cannot be saved to the '{metadata_file}' file. See the nested exception for details."
            )

        try:
            self._file_system.make_directory(metadata_file.parent)
        except OSError as ex:
            raise save_error() from ex

        try:
            self._yaml_dumper.dump(metadata, metadata_file)
        except OSError as ex:
            raise save_error() from ex


@final
class FileCheckpointMetadataLoader:
    """Provides checkpoint model metadata saved by a :class:`FileCheckpointManager.`"""

    _checkpoint_dir: Path
    _file_system: FileSystem

    def __init__(
        self,
        checkpoint_dir: Path,
        file_system: FileSystem,
        metadata_file_loader: AssetMetadataFileLoader,
    ) -> None:
        """
        :param checkpoint_dir:
            The base directory under which the checkpoints are stored.
        """
        super().__init__()

        self._checkpoint_dir = checkpoint_dir
        self._file_system = file_system
        self._metadata_file_loader = metadata_file_loader

    def load(self) -> AssetMetadataProvider:
        cache = self._load_cache()

        return CachedAssetMetadataProvider(cache)

    def _load_cache(self) -> dict[str, dict[str, object]]:
        cache: dict[str, dict[str, object]] = {}

        metadata_file = self._checkpoint_dir.joinpath("model.yaml")

        for name, metadata in self._metadata_file_loader.load(metadata_file):
            cache[name] = metadata

        try:
            metadata = cache["checkpoint@"]
        except KeyError:
            raise AssetMetadataLoadError(
                "The checkpoint metadata does not have a 'checkpoint@' entry."
            ) from None

        def add_checkpoint_metadata(name: str, step_nr: int) -> None:
            model_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}/model")

            cache[name] = {"base": "checkpoint", "checkpoint": str(model_dir)}

        max_step_nr = -1

        scores = []

        def iter_step_dirs() -> Iterable[Path]:
            try:
                for step_dir in self._file_system.glob(self._checkpoint_dir, "step_*"):
                    if not self._file_system.is_dir(step_dir):
                        continue

                    yield step_dir
            except OSError as ex:
                raise AssetMetadataLoadError(
                    f"The '{self._checkpoint_dir}' base checkpoint directory cannot be traversed. See the nested exception for details."
                ) from ex

        for step_dir in iter_step_dirs():
            try:
                step_nr = int(step_dir.name[5:])
            except ValueError:
                continue

            add_checkpoint_metadata(f"checkpoint_step_{step_nr}@", step_nr)

            max_step_nr = max(max_step_nr, step_nr)

            # Load score.
            score_file = self._checkpoint_dir.joinpath(f"scores/step_{step_nr}.txt")

            def load_error() -> AssetMetadataLoadError:
                return AssetMetadataLoadError(
                    f"The score of the training step {step_nr} cannot be loaded from the '{score_file}' file. See the nested exception for details."
                )

            try:
                fp = self._file_system.open_text(score_file)
            except FileNotFoundError:
                fp = None
            except OSError as ex:
                raise load_error() from ex

            if fp is None:
                continue

            try:
                line = fp.readline()
            except OSError as ex:
                raise load_error() from ex
            finally:
                fp.close()

            try:
                score = float(line)
            except ValueError:
                raise AssetMetadataLoadError(
                    f"The score of the training step {step_nr} cannot be parsed as a floating-point number."
                ) from None

            scores.append((score, step_nr))

        if max_step_nr == -1:
            return cache

        add_checkpoint_metadata("last_checkpoint@", max_step_nr)

        if not scores:
            return cache

        scores.sort(reverse=True)

        best_step_nr = scores[0][1]

        add_checkpoint_metadata("best_checkpoint@", best_step_nr)

        for idx, (_, step_nr) in enumerate(scores):
            add_checkpoint_metadata(f"best_checkpoint_{idx}@", step_nr)

        return cache
