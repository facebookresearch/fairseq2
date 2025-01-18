# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import final

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

        def add_checkpoint_metadata(name: str, path: Path) -> None:
            cache[name] = {"base": "checkpoint", "checkpoint": str(path)}

        max_step_nr = -1

        scores = []

        try:
            for step_dir in self._file_system.glob(self._checkpoint_dir, "step_*"):
                if not self._file_system.is_dir(step_dir):
                    continue

                try:
                    step_nr = int(step_dir.name[5:])
                except ValueError:
                    continue

                add_checkpoint_metadata(
                    f"checkpoint_step_{step_nr}@", step_dir.joinpath(filename)
                )

                max_step_nr = max(max_step_nr, step_nr)

                # Load score.
                score_file = step_dir.joinpath("score.txt")
                if self._file_system.exists(score_file):
                    fp = self._file_system.open_text(score_file)

                    try:
                        line = fp.readline()
                    except OSError as ex:
                        raise AssetMetadataError(
                            f"The score of the training step {step_nr} cannot be loaded from the '{score_file}' file. See the nested exception for details."
                        ) from ex
                    finally:
                        fp.close()

                    try:
                        score = float(line)
                    except ValueError:
                        raise AssetMetadataError(
                            f"The score of the training step {step_nr} cannot be parsed as a floating-point number."
                        ) from None

                    scores.append((score, step_nr))
        except OSError as ex:
            raise AssetMetadataError(
                f"The base '{self._checkpoint_dir}' checkpoint directory cannot be traversed. See the nested exception for details."
            ) from ex

        if max_step_nr >= 0:
            last_model_file = self._checkpoint_dir.joinpath(
                f"step_{max_step_nr}/{filename}"
            )

            add_checkpoint_metadata("last_checkpoint@", last_model_file)

        scores.sort()

        last_idx = len(scores) - 1

        for i, (_, step_nr) in enumerate(scores):
            model_file = self._checkpoint_dir.joinpath(f"step_{step_nr}/{filename}")

            add_checkpoint_metadata(f"checkpoint_lowest_{i}@", model_file)
            add_checkpoint_metadata(f"checkpoint_highest_{last_idx - i}@", model_file)

    def _load_tokenizer(self, cache: dict[str, dict[str, object]]) -> None:
        metadata_file = self._checkpoint_dir.joinpath("tokenizer.yaml")
        if self._file_system.exists(metadata_file):
            for name, metadata in self._metadata_file_loader.load(metadata_file):
                cache[name] = metadata
