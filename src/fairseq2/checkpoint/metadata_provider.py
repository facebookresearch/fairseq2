# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias, final

from typing_extensions import override

from fairseq2.assets import (
    AbstractAssetMetadataProvider,
    AssetMetadataError,
    load_metadata_file,
)
from fairseq2.utils.yaml import load_yaml


@final
class FileCheckpointMetadataProvider(AbstractAssetMetadataProvider):
    """Provides checkpoint model metadata saved by a :class:`FileCheckpointManager.`"""

    _checkpoint_dir: Path

    def __init__(self, checkpoint_dir: Path) -> None:
        """
        :param checkpoint_dir:
            The base directory under which the checkpoints are stored.
        """
        super().__init__()

        self._checkpoint_dir = checkpoint_dir

    @override
    def _load_cache(self) -> dict[str, dict[str, object]]:
        cache: dict[str, dict[str, object]] = {}

        self._load_model(cache)

        self._load_tokenizer(cache)

        return cache

    def _load_model(self, cache: dict[str, dict[str, object]]) -> None:
        checkpoint_dir = self._checkpoint_dir.expanduser().resolve()

        metadata_file = checkpoint_dir.joinpath("model.yaml")

        for name, metadata in load_metadata_file(metadata_file, load_yaml):
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
            for step_dir in checkpoint_dir.glob("step_*"):
                if not step_dir.is_dir():
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
                if score_file.exists():
                    try:
                        with score_file.open() as fp:
                            line = fp.readline()
                    except OSError as ex:
                        raise AssetMetadataError(
                            f"The score of the training step {step_nr} cannot be loaded from the '{score_file}' file. See the nested exception for details."
                        ) from ex

                    try:
                        score = float(line)
                    except ValueError:
                        raise AssetMetadataError(
                            f"The score of the training step {step_nr} cannot be parsed as a floating-point number."
                        ) from None

                    scores.append((score, step_nr))
        except OSError as ex:
            raise AssetMetadataError(
                f"The base '{checkpoint_dir}' checkpoint directory cannot be traversed. See the nested exception for details."
            ) from ex

        if max_step_nr >= 0:
            last_model_file = checkpoint_dir.joinpath(f"step_{max_step_nr}/{filename}")

            add_checkpoint_metadata("last_checkpoint@", last_model_file)

        scores.sort()

        last_idx = len(scores) - 1

        for i, (_, step_nr) in enumerate(scores):
            model_file = checkpoint_dir.joinpath(f"step_{step_nr}/{filename}")

            add_checkpoint_metadata(f"checkpoint_lowest_{i}@", model_file)
            add_checkpoint_metadata(f"checkpoint_highest_{last_idx - i}@", model_file)

    def _load_tokenizer(self, cache: dict[str, dict[str, object]]) -> None:
        checkpoint_dir = self._checkpoint_dir.expanduser().resolve()

        metadata_file = checkpoint_dir.joinpath("tokenizer.yaml")
        if metadata_file.exists():
            for name, metadata in load_metadata_file(metadata_file, load_yaml):
                cache[name] = metadata


CheckpointModelMetadataProvider: TypeAlias = FileCheckpointMetadataProvider  # compat
