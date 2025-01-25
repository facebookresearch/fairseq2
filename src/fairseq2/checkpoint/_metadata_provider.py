# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, final

from typing_extensions import override

from fairseq2.assets import (
    AbstractAssetMetadataProvider,
    AssetMetadataError,
    MetadataFileLoader,
)
from fairseq2.utils.file import FileSystem


@final
class FileCheckpointMetadataProvider(AbstractAssetMetadataProvider):
    """Provides checkpoint model metadata saved by a :class:`FileCheckpointManager.`"""

    _checkpoint_dir: Path
    _file_system: FileSystem

    def __init__(
        self,
        checkpoint_dir: Path,
        file_system: FileSystem,
        metadata_file_loader: MetadataFileLoader,
    ) -> None:
        """
        :param checkpoint_dir:
            The base directory under which the checkpoints are stored.
        """
        super().__init__()

        self._checkpoint_dir = checkpoint_dir
        self._file_system = file_system
        self._metadata_file_loader = metadata_file_loader

    @override
    def _load_cache(self) -> dict[str, dict[str, object]]:
        cache: dict[str, dict[str, object]] = {}

        self._load_model(cache)

        self._load_tokenizer(cache)

        return cache

    def _load_model(self, cache: dict[str, dict[str, object]]) -> None:
        metadata_file = self._checkpoint_dir.joinpath("model.yaml")

        for name, metadata in self._metadata_file_loader.load(metadata_file):
            cache[name] = metadata

        try:
            metadata = cache["checkpoint@"]
        except KeyError:
            raise AssetMetadataError(
                "The checkpoint metadata does not have a 'checkpoint@' entry."
            ) from None

        num_shards = metadata.get("num_shards", 1)

        if not isinstance(num_shards, int) or num_shards < 1:
            raise AssetMetadataError(
                "The 'num_shards' value in the checkpoint metadata is not a positive integer."
            )

        if num_shards == 1:
            filename = "model.pt"
        else:
            filename = "model.{shard_idx}.pt"

        def add_checkpoint_metadata(name: str, step_nr: int) -> None:
            model_file = self._checkpoint_dir.joinpath(f"step_{step_nr}/{filename}")

            cache[name] = {"base": "checkpoint", "checkpoint": str(model_file)}

        max_step_nr = -1

        scores = []

        def iter_step_dirs() -> Iterable[Path]:
            try:
                for step_dir in self._file_system.glob(self._checkpoint_dir, "step_*"):
                    if not self._file_system.is_dir(step_dir):
                        continue

                    yield step_dir
            except OSError as ex:
                raise AssetMetadataError(
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
            score_file = step_dir.joinpath("score.txt")

            def load_error() -> AssetMetadataError:
                return AssetMetadataError(
                    f"The score of the training step {step_nr} cannot be loaded from the '{score_file}' file. See the nested exception for details."
                )

            try:
                score_exists = self._file_system.exists(score_file)
            except OSError as ex:
                raise load_error() from ex

            if score_exists:
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
                    raise AssetMetadataError(
                        f"The score of the training step {step_nr} cannot be parsed as a floating-point number."
                    ) from None

                scores.append((score, step_nr))

        if max_step_nr == -1:
            return

        add_checkpoint_metadata("last_checkpoint@", max_step_nr)

        if not scores:
            return

        scores.sort()

        best_step_nr = scores[-1][1]

        add_checkpoint_metadata("best_checkpoint@", best_step_nr)

        for idx, (_, step_nr) in enumerate(reversed(scores)):
            add_checkpoint_metadata(f"best_checkpoint_{idx}@", step_nr)

    def _load_tokenizer(self, cache: dict[str, dict[str, object]]) -> None:
        metadata_file = self._checkpoint_dir.joinpath("tokenizer.yaml")

        try:
            tokenizer_exists = self._file_system.exists(metadata_file)
        except OSError as ex:
            raise AssetMetadataError(
                f"The '{metadata_file}' path cannot be accessed. See the nested exception for details."
            ) from ex

        if tokenizer_exists:
            for name, metadata in self._metadata_file_loader.load(metadata_file):
                cache[name] = metadata
