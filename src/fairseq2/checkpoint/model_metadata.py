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
    AssetMetadataProvider,
    AssetMetadataSource,
    AssetMetadataSourceNotFoundError,
    BadAssetMetadataError,
    BadAssetMetadataFileError,
    CachedAssetMetadataProvider,
    _AssetMetadataFileLoader,
)
from fairseq2.error import InternalError
from fairseq2.file_system import FileSystem
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.yaml import YamlDumper, YamlError


class _ModelMetadataDumper(ABC):
    @abstractmethod
    def dump(self, checkpoint_dir: Path, family_name: str, config: object) -> None:
        """
        :raises ValueError:
        """


@final
class _StandardModelMetadataDumper(_ModelMetadataDumper):
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
        try:
            unstructured_config = self._value_converter.unstructure(config)
        except StructureError as ex:
            raise ValueError("failed to unstructure `config`") from ex

        metadata: dict[str, object] = {
            "name": "checkpoint",
            "model_family": family_name,
            "model_config": unstructured_config,
        }

        try:
            self._file_system.make_directory(checkpoint_dir)
        except OSError as ex:
            raise XXX(f"failed to create directory '{checkpoint_dir}'") from ex

        metadata_file = checkpoint_dir.joinpath("model.yaml")

        try:
            self._yaml_dumper.dump(metadata, metadata_file)
        except YamlError as ex:
            raise InternalError(
                "failed to serialize model configuration to YAML"
            ) from ex
        except OSError as ex:
            raise XXX(f"failed to write file '{metadata_file}'") from ex


@final
class _ModelMetadataSource(AssetMetadataSource):
    def __init__(
        self, checkpoint_dir: Path, metadata_loader: _ModelMetadataLoader
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._metadata_loader = metadata_loader

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        source = f"checkpoint:{self._checkpoint_dir}"

        yield self._metadata_loader.load(source, self._checkpoint_dir)


class _ModelMetadataLoader(ABC):
    @abstractmethod
    def load(self, source: str, checkpoint_dir: Path) -> AssetMetadataProvider:
        """
        :raises AssetMetadataSourceNotFoundError:
        :raises BadAssetMetadataError:
        :raises AssetMetadataError:
        """


@final
class _StandardModelMetadataLoader(_ModelMetadataLoader):
    def __init__(
        self,
        file_system: FileSystem,
        metadata_file_loader: _AssetMetadataFileLoader,
    ) -> None:
        self._file_system = file_system
        self._metadata_file_loader = metadata_file_loader

    @override
    def load(self, source: str, checkpoint_dir: Path) -> AssetMetadataProvider:
        metadata = {}

        fs = self._file_system

        try:
            checkpoint_dir = fs.resolve(checkpoint_dir)
        except RuntimeError as ex:
            raise AssetMetadataError(
                source, f"failed to access path '{checkpoint_dir}'"
            ) from ex

        try:
            dir_exists = fs.exists(checkpoint_dir)
        except OSError as ex:
            raise AssetMetadataError(
                source, f"failed to access directory '{checkpoint_dir}'"
            ) from ex

        if not dir_exists:
            raise AssetMetadataSourceNotFoundError(source)

        file = checkpoint_dir.joinpath("model.yaml")

        try:
            file_metadata = self._metadata_file_loader.load(file)
        except BadAssetMetadataFileError as ex:
            raise BadAssetMetadataError(
                source, f"failed to load model metadata file '{file}'"
            ) from ex
        except OSError as ex:
            raise AssetMetadataError(source, f"failed to read file '{file}'") from ex

        for name, asset_metadata in file_metadata:
            metadata[name] = asset_metadata

        if "checkpoint@" not in metadata:
            raise BadAssetMetadataError(
                source, f"file '{file}' does not have an asset named 'checkpoint'."
            )

        def add_metadata(name: str, step_nr: int) -> None:
            model_dir = checkpoint_dir.joinpath(f"step_{step_nr}/model")

            metadata[name] = {"base": "checkpoint", "checkpoint": str(model_dir)}

        max_step_nr = -1

        scores = []

        def step_dirs() -> Iterator[Path]:
            try:
                yield from fs.glob(checkpoint_dir, "step_*")
            except OSError as ex:
                raise AssetMetadataError(
                    source, f"failed to glob directory '{checkpoint_dir}'"
                ) from ex

        for step_dir in step_dirs():
            try:
                dir_exists = fs.exists(step_dir)
            except OSError as ex:
                raise AssetMetadataError(
                    source, f"failed to access directory '{step_dir}'"
                ) from ex

            if not dir_exists:
                continue

            try:
                step_nr = int(step_dir.name[5:])
            except ValueError:
                continue

            add_metadata(f"checkpoint_step_{step_nr}@", step_nr)

            max_step_nr = max(max_step_nr, step_nr)

            # Load score.
            score_file = checkpoint_dir.joinpath(f"scores/step_{step_nr}.txt")

            try:
                with fs.open_text(score_file) as fp:
                    line = fp.readline()
            except FileNotFoundError:
                continue
            except OSError as ex:
                raise AssetMetadataError(
                    source, f"failed to read file '{score_file}'"
                ) from ex

            try:
                score = float(line)
            except ValueError:
                raise BadAssetMetadataError(
                    source, f"score of step {step_nr} in file '{score_file}' is not a floating-point number"  # fmt: skip
                ) from None

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
