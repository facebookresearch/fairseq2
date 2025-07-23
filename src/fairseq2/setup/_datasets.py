# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.context import RuntimeContext
from fairseq2.datasets import DatasetHandler, DatasetLoader, StandardDatasetHandler
from fairseq2.datasets.asr import GENERIC_ASR_DATASET_FAMILY, GenericAsrDataset
from fairseq2.datasets.instruction import (
    GENERIC_INSTRUCTION_DATASET_FAMILY,
    GenericInstructionDataset,
)
from fairseq2.datasets.jsonl import JSONL_DATASET_FAMILY, JsonlDataset
from fairseq2.datasets.parallel_text import (
    GENERIC_PARALLEL_TEXT_DATASET_FAMILY,
    GenericParallelTextDataset,
)
from fairseq2.datasets.preference import (
    GENERIC_PREFERENCE_DATASET_FAMILY,
    GenericPreferenceDataset,
)
from fairseq2.datasets.prompt import (
    GENERIC_PROMPT_DATASET_FAMILY,
    GenericPromptDataset,
)
from fairseq2.datasets.speech import GENERIC_SPEECH_DATASET_FAMILY, GenericSpeechDataset
from fairseq2.datasets.text import GENERIC_TEXT_DATASET_FAMILY, GenericTextDataset
from fairseq2.datasets.text_parquet import (
    PARQUET_TEXT_DATASET_FAMILY,
    ParquetTextDataset,
)
from fairseq2.registry import Registry


def _register_dataset_families(context: RuntimeContext) -> None:
    # fmt: off
    registrar = DatasetRegistrar(context)

    registrar.register_family(
        GENERIC_ASR_DATASET_FAMILY,
        GenericAsrDataset,
        GenericAsrDataset.from_path,
    )

    registrar.register_family(
        GENERIC_INSTRUCTION_DATASET_FAMILY,
        GenericInstructionDataset,
        GenericInstructionDataset.from_path,
    )

    registrar.register_family(
        GENERIC_PARALLEL_TEXT_DATASET_FAMILY,
        GenericParallelTextDataset,
        GenericParallelTextDataset.from_path,
    )

    registrar.register_family(
        GENERIC_PREFERENCE_DATASET_FAMILY,
        GenericPreferenceDataset,
        GenericPreferenceDataset.from_path,
    )

    registrar.register_family(
        GENERIC_PROMPT_DATASET_FAMILY,
        GenericPromptDataset,
        GenericPromptDataset.from_path,
    )

    registrar.register_family(
        GENERIC_SPEECH_DATASET_FAMILY,
        GenericSpeechDataset,
        GenericSpeechDataset.from_path,
    )

    registrar.register_family(
        GENERIC_TEXT_DATASET_FAMILY,
        GenericTextDataset,
        GenericTextDataset.from_path,
    )

    registrar.register_family(
        JSONL_DATASET_FAMILY,
        JsonlDataset,
        JsonlDataset.from_path,
    )
    registrar.register_family(
        PARQUET_TEXT_DATASET_FAMILY,
        ParquetTextDataset,
        ParquetTextDataset.from_path,
    )
    # fmt: on


@final
class DatasetRegistrar:
    _context: RuntimeContext
    _registry: Registry[DatasetHandler]

    def __init__(self, context: RuntimeContext) -> None:
        self._context = context

        self._registry = context.get_registry(DatasetHandler)

    def register_family(
        self, family: str, kls: type[object], loader: DatasetLoader
    ) -> None:
        file_system = self._context.file_system

        asset_download_manager = self._context.asset_download_manager

        handler = StandardDatasetHandler(
            family, kls, loader, file_system, asset_download_manager
        )

        self._registry.register(family, handler)
