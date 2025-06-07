# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, final

from fairseq2.assets import (
    AssetMetadataError,
    AssetMetadataFileLoader,
    AssetMetadataProvider,
    CachedAssetMetadataProvider,
    YamlAssetMetadataFileLoader,
)
from fairseq2.error import InfraError
from fairseq2.file_system import FileSystem
from fairseq2.gang import Gangs
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.utils.structured import ValueConverter
from fairseq2.utils.yaml import YamlDumper, YamlLoader


@final
class CheckpointAssetMetadataSaver:
    _file_system: FileSystem
    _yaml_dumper: YamlDumper
    _value_converter: ValueConverter

    def __init__(
        self,
        file_system: FileSystem,
        yaml_dumper: YamlDumper,
        value_converter: ValueConverter,
    ) -> None:
        self._file_system = file_system
        self._yaml_dumper = yaml_dumper
        self._value_converter = value_converter

    def save(
        self,
        checkpoint_dir: Path,
        gangs: Gangs,
        model_family: str,
        model_config: object,
    ) -> None:
        if gangs.root.rank == 0:
            self._save_asset_card(checkpoint_dir, model_family, model_config)

        gangs.root.barrier()

    def _save_asset_card(
        self, checkpoint_dir: Path, model_family: str, model_config: object
    ) -> None:
        model_config = self._value_converter.unstructure(model_config)

        metadata: dict[str, object] = {
            "name": "checkpoint",
            "model_family": model_family,
            "model_config": {
                "_set_": model_config,
            },
        }

        model_file = checkpoint_dir.joinpath("model.yaml")

        try:
            self._file_system.make_directory(model_file.parent)
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while creating the '{model_file.parent}' directory. See the nested exception for details."
            ) from ex

        try:
            self._yaml_dumper.dump(metadata, model_file)
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while saving the checkpoint metadata to the '{model_file}' file. See the nested exception for details."
            ) from ex


@final
class CheckpointAssetMetadataLoader:
    """Provides checkpoint model metadata saved by a :class:`FileCheckpointManager.`"""

    _file_system: FileSystem
    _metadata_file_loader: AssetMetadataFileLoader

    def __init__(
        self,
        file_system: FileSystem,
        metadata_file_loader: AssetMetadataFileLoader,
    ) -> None:
        super().__init__()

        self._file_system = file_system
        self._metadata_file_loader = metadata_file_loader

    def load(self, checkpoint_dir: Path) -> AssetMetadataProvider:
        source = f"checkpoint:{checkpoint_dir}"

        metadata = {}

        model_file = checkpoint_dir.joinpath("model.yaml")

        for name, asset_metadata in self._metadata_file_loader.load(model_file, source):
            metadata[name] = asset_metadata

        if "checkpoint@" not in metadata:
            raise AssetMetadataError(
                source, "The checkpoint does not have an asset named 'checkpoint'."  # fmt: skip
            )

        def add_checkpoint_metadata(name: str, step_nr: int) -> None:
            model_dir = checkpoint_dir.joinpath(f"step_{step_nr}/model")

            metadata[name] = {"base": "checkpoint", "checkpoint": str(model_dir)}

        max_step_nr = -1

        scores = []

        def iter_step_dirs() -> Iterable[Path]:
            try:
                for step_dir in self._file_system.glob(checkpoint_dir, "step_*"):
                    if not self._file_system.is_dir(step_dir):
                        continue

                    yield step_dir
            except OSError as ex:
                raise InfraError(
                    f"A system error has occurred while traversing the '{checkpoint_dir}' checkpoint directory. See the nested exception for details."
                ) from ex

        for step_dir in iter_step_dirs():
            try:
                step_nr = int(step_dir.name[5:])
            except ValueError:
                continue

            add_checkpoint_metadata(f"checkpoint_step_{step_nr}@", step_nr)

            max_step_nr = max(max_step_nr, step_nr)

            # Load score.
            score_file = checkpoint_dir.joinpath(f"scores/step_{step_nr}.txt")

            try:
                fp = self._file_system.open_text(score_file)
            except FileNotFoundError:
                fp = None
            except OSError as ex:
                raise InfraError(
                    f"A system error has occurred while loading the score of the training step {step_nr} from the '{score_file}' file. See the nested exception for details."
                ) from ex

            if fp is None:
                continue

            try:
                line = fp.readline()
            except OSError as ex:
                raise InfraError(
                    f"A system error has occurred while loading the score of the training step {step_nr} from the '{score_file}' file. See the nested exception for details."
                ) from ex
            finally:
                fp.close()

            try:
                score = float(line)
            except ValueError as ex:
                raise AssetMetadataError(
                    source, f"The score of the training step {step_nr} cannot be parsed as a floating-point number. See the nested exception for details."  # fmt: skip
                ) from ex

            scores.append((score, step_nr))

        if max_step_nr >= 0:
            add_checkpoint_metadata("last_checkpoint@", max_step_nr)

        if scores:
            scores.sort(reverse=True)

            best_step_nr = scores[0][1]

            add_checkpoint_metadata("best_checkpoint@", best_step_nr)

            for idx, (_, step_nr) in enumerate(scores):
                add_checkpoint_metadata(f"best_checkpoint_{idx}@", step_nr)

        return CachedAssetMetadataProvider(source, metadata)


def register_checkpoint_assets(
    container: DependencyContainer, checkpoint_dir: Path, *, not_exist_ok: bool = False
) -> None:
    def load_assets(resolver: DependencyResolver) -> AssetMetadataProvider | None:
        file_system = resolver.resolve(FileSystem)

        yaml_loader = resolver.resolve(YamlLoader)

        metadata_file_loader = YamlAssetMetadataFileLoader(yaml_loader)

        metadata_loader = CheckpointAssetMetadataLoader(
            file_system, metadata_file_loader
        )

        try:
            return metadata_loader.load(checkpoint_dir)
        except FileNotFoundError as ex:
            if not_exist_ok:
                return None

            raise AssetMetadataError(
                f"checkpoint:{checkpoint_dir}", f"The '{checkpoint_dir}' checkpoint directory is not found."  # fmt: skip
            ) from ex

    container.register(AssetMetadataProvider, load_assets)
