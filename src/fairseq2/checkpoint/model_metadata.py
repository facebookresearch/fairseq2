# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, final

from typing_extensions import override

from fairseq2.assets import (
    AssetMetadataError,
    AssetMetadataFileLoader,
    AssetMetadataProvider,
    AssetMetadataSource,
    CachedAssetMetadataProvider,
)
from fairseq2.error import raise_operational_system_error
from fairseq2.file_system import FileSystem
from fairseq2.utils.structured import ValueConverter
from fairseq2.utils.yaml import YamlDumper


class ModelMetadataDumper(ABC):
    @abstractmethod
    def dump(self, checkpoint_dir: Path, family_name: str, config: object) -> None: ...


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
    def dump(self, checkpoint_dir: Path, family_name: str, config: object) -> None:
        unstructured_config = self._value_converter.unstructure(config)

        metadata: dict[str, object] = {
            "name": "checkpoint",
            "model_family": family_name,
            "model_config": unstructured_config,
        }

        self._file_system.make_directory(checkpoint_dir)

        metadata_file = checkpoint_dir.joinpath("model.yaml")

        self._yaml_dumper.dump(metadata, metadata_file)


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

        def iter_step_dirs() -> Iterator[Path]:
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
            except ValueError:
                msg = f"Score of the training step {step_nr} cannot be parsed as a floating-point number."

                raise AssetMetadataError(source, msg) from None

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


@final
class ModelMetadataSource(AssetMetadataSource):
    def __init__(
        self, checkpoint_dir: Path, metadata_loader: ModelMetadataLoader
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._metadata_loader = metadata_loader

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        try:
            yield self._metadata_loader.load(self._checkpoint_dir)
        except OSError as ex:
            raise_operational_system_error(ex)
