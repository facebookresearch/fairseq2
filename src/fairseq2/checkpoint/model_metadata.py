# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, final

from typing_extensions import override

from fairseq2.assets import (
    AssetMetadataError,
    AssetMetadataFileLoader,
    AssetMetadataProvider,
    CachedAssetMetadataProvider,
)
from fairseq2.error import InternalError, raise_operational_system_error
from fairseq2.file_system import FileSystem
from fairseq2.model import Model
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.utils.structured import ValueConverter
from fairseq2.utils.yaml import YamlDumper, YamlError


class ModelMetadataDumper(ABC):
    @abstractmethod
    def dump(self, checkpoint_dir: Path, model: Model) -> None: ...


@final
class StandardModelMetadataDumper(ModelMetadataDumper):
    def __init__(
        self,
        file_system: FileSystem,
        yaml_dumper: YamlDumper,
        value_converter: ValueConverter,
    ) -> None:
        self._file_system = file_system
        self._yaml_dumper = yaml_dumper
        self._value_converter = value_converter

    @override
    def dump(self, checkpoint_dir: Path, model: Model) -> None:
        config = self._value_converter.unstructure(model.config)

        metadata: dict[str, object] = {
            "name": "checkpoint",
            "model_family": model.handler.family,
            "model_config": {
                "_set_": config,
            },
        }

        self._file_system.make_directory(checkpoint_dir)

        file = checkpoint_dir.joinpath("model.yaml")

        try:
            self._yaml_dumper.dump(metadata, file)
        except YamlError as ex:
            raise InternalError(
                "Checkpoint model metadata cannot be saved as YAML."
            ) from ex


class ModelMetadataLoader(ABC):
    @abstractmethod
    def load(self, checkpoint_dir: Path) -> AssetMetadataProvider: ...


@final
class StandardModelMetadataLoader(ModelMetadataLoader):
    def __init__(
        self,
        file_system: FileSystem,
        metadata_file_loader: AssetMetadataFileLoader,
    ) -> None:
        self._file_system = file_system
        self._metadata_file_loader = metadata_file_loader

    @override
    def load(self, checkpoint_dir: Path) -> AssetMetadataProvider:
        source = f"checkpoint:{checkpoint_dir}"

        metadata = {}

        checkpoint_dir = self._file_system.resolve(checkpoint_dir)

        file = checkpoint_dir.joinpath("model.yaml")

        for name, asset_metadata in self._metadata_file_loader.load(file, source):
            metadata[name] = asset_metadata

        if "checkpoint@" not in metadata:
            msg = f"{checkpoint_dir} checkpoint directory does not have an asset named checkpoint."

            raise AssetMetadataError(source, msg)

        def add_metadata(name: str, step_nr: int) -> None:
            model_dir = checkpoint_dir.joinpath(f"step_{step_nr}/model")

            metadata[name] = {"base": "checkpoint", "checkpoint": str(model_dir)}

        max_step_nr = -1

        scores = []

        def iter_step_dirs() -> Iterable[Path]:
            for step_dir in self._file_system.glob(checkpoint_dir, "step_*"):
                if not self._file_system.is_dir(step_dir):
                    continue

                yield step_dir

        for step_dir in iter_step_dirs():
            try:
                step_nr = int(step_dir.name[5:])
            except ValueError:
                continue

            add_metadata(f"checkpoint_step_{step_nr}@", step_nr)

            max_step_nr = max(max_step_nr, step_nr)

            # Load score.
            score_file = checkpoint_dir.joinpath(f"scores/step_{step_nr}.txt")

            try:
                fp = self._file_system.open_text(score_file)
            except FileNotFoundError:
                fp = None

            if fp is None:
                continue

            try:
                line = fp.readline()
            finally:
                fp.close()

            try:
                score = float(line)
            except ValueError as ex:
                msg = f"Score of the training step {step_nr} cannot be parsed as a floating-point number."

                raise AssetMetadataError(source, msg) from ex

            scores.append((score, step_nr))

        if max_step_nr >= 0:
            add_metadata("last_checkpoint@", max_step_nr)

        if scores:
            scores.sort(reverse=True)

            best_step_nr = scores[0][1]

            add_metadata("best_checkpoint@", best_step_nr)

            for idx, (_, step_nr) in enumerate(scores):
                add_metadata(f"best_checkpoint_{idx}@", step_nr)

        return CachedAssetMetadataProvider(source, metadata)


def register_checkpoint_models(
    container: DependencyContainer, checkpoint_dir: Path
) -> None:
    def load_metadata(resolver: DependencyResolver) -> AssetMetadataProvider:
        metadata_loader = resolver.resolve(ModelMetadataLoader)

        source = ModelMetadataSource(checkpoint_dir, metadata_loader)

        return source.load()

    container.register(AssetMetadataProvider, load_metadata)


@final
class ModelMetadataSource:
    def __init__(
        self, checkpoint_dir: Path, metadata_loader: ModelMetadataLoader
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._metadata_loader = metadata_loader

    def load(self) -> AssetMetadataProvider:
        try:
            return self._metadata_loader.load(self._checkpoint_dir)
        except OSError as ex:
            raise_operational_system_error(ex)
