# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder, read_sequence
from fairseq2.datasets import DataPipelineReader, SyncMode
from fairseq2.gang import Gangs

npc = 10

MULTIMODAL_GENERATE_DATASET_FAMILY: Final = "multimodal_generate"


class MultimodalGenerateDataset:
    def __init__(self, files: Sequence[Path]) -> None:
        self._files = [f.expanduser().resolve() for f in files]

    def create_reader(
        self, gangs: Gangs, *, batch_size: int, prefetch: int = 1
    ) -> DataPipelineReader[list[dict[str, Any]]]:
        if len(self._files) == 1:
            builder = self._read_jsonl(self._files[0])
        else:
            pipelines = []
            for file in self._files:
                pipeline = self._read_jsonl(file).and_return()
                pipelines.append(pipeline)
            builder = DataPipeline.concat(pipelines)

        # Shard across data-parallel ranks.
        builder.shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)

        # Bucket into batches (list of dicts).
        builder.bucket(batch_size, drop_remainder=False)

        # Prefetch batches in background.
        builder.prefetch(prefetch)

        pipeline = builder.and_return()

        return DataPipelineReader[list[dict[str, Any]]](
            pipeline, gangs, sync_mode=SyncMode.UNTIL_LAST
        )

    def _read_jsonl(self, path: Path) -> DataPipelineBuilder:
        lines: list[str] = []
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                lines.append(line)
        return read_sequence(lines).map(json.loads, num_parallel_calls=npc)


@dataclass
class MultimodalGenerateDatasetConfig:
    paths: list[Path] = field(default_factory=list)


def open_multimodal_generate_dataset(
    config: MultimodalGenerateDatasetConfig,
) -> MultimodalGenerateDataset:
    return MultimodalGenerateDataset(config.paths)
