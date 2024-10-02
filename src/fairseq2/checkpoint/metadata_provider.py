# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, final

from typing_extensions import override

from fairseq2.assets.metadata_provider import (
    AbstractAssetMetadataProvider,
    AssetMetadataError,
    load_metadata_file,
)


@final
class FileCheckpointMetadataProvider(AbstractAssetMetadataProvider):
    """Provides checkpoint model metadata saved by a :class:`FileCheckpointManager.`"""

    _checkpoint_dir: Path
    _lower_score_better: bool

    def __init__(
        self, checkpoint_dir: Path, *, lower_score_better: bool = False
    ) -> None:
        """
        :param checkpoint_dir:
            The base directory under which the checkpoints are stored.
        :param lower_score_better:
            If ``True``, lower scores are considered better.
        """
        super().__init__()

        self._checkpoint_dir = checkpoint_dir.expanduser().resolve()

        self._lower_score_better = lower_score_better

    @override
    def _load_cache(self) -> dict[str, dict[str, Any]]:
        metadata_file = self._checkpoint_dir.joinpath("model.yaml")
        if not metadata_file.exists():
            raise AssetMetadataError(
                f"The checkpoint model metadata (model.yaml) cannot be found under {self._checkpoint_dir}. Make sure that the specified directory is the *base* checkpoint directory used during training (i.e. directory passed to `FileCheckpointManager.save_model_metadata()`)."
            )

        cache = dict(load_metadata_file(metadata_file))

        try:
            metadata = cache["checkpoint@"]
        except KeyError as ex:
            raise AssetMetadataError(
                "The checkpoint model metadata has an invalid format."
            ) from ex

        try:
            num_shards = int(metadata["num_shards"])
        except KeyError:
            num_shards = 1
        except ValueError as ex:
            raise AssetMetadataError(
                "The checkpoint model metadata has an invalid format."
            ) from ex

        if num_shards == 1:
            filename = "model.pt"
        else:
            filename = "model.{shard_idx}.pt"

        def add_checkpoint_metadata(name: str, path: Path) -> None:
            cache[name] = {"base": "checkpoint", "checkpoint": str(path)}

        max_step_nr = -1

        scores = []

        try:
            for step_dir in self._checkpoint_dir.glob("step_*"):
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
                        raise RuntimeError(
                            f"The score of training step {step_nr} cannot be loaded. See nested exception for details."
                        ) from ex

                    try:
                        score = float(line)
                    except ValueError as ex:
                        raise RuntimeError(
                            f"The score of training step {step_nr} cannot be loaded. See nested exception for details."
                        ) from ex

                    scores.append((score, step_nr))
        except OSError as ex:
            raise RuntimeError(
                "The base checkpoint directory cannot be traversed. See nested exception for details."
            ) from ex

        if max_step_nr >= 0:
            last_model_file = self._checkpoint_dir.joinpath(
                f"step_{max_step_nr}/{filename}"
            )

            add_checkpoint_metadata("last_checkpoint@", last_model_file)

        if self._lower_score_better:
            scores.sort(key=lambda e: (-e[0], e[1]))
        else:
            scores.sort()

        for rank, (_, step_nr) in enumerate(reversed(scores)):
            model_file = self._checkpoint_dir.joinpath(f"step_{step_nr}/{filename}")

            add_checkpoint_metadata(f"checkpoint_best_{rank}@", model_file)

        return cache


# compat
CheckpointModelMetadataProvider = FileCheckpointMetadataProvider
