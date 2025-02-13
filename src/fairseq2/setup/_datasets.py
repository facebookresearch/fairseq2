# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.datasets import DatasetHandler, StandardDatasetHandler
from fairseq2.datasets.asr import GENERIC_ASR_DATASET_FAMILY, GenericAsrDataset
from fairseq2.datasets.instruction import (
    GENERIC_INSTRUCTION_DATASET_FAMILY,
    GenericInstructionDataset,
)
from fairseq2.datasets.parallel_text import (
    GENERIC_PARALLEL_TEXT_DATASET_FAMILY,
    GenericParallelTextDataset,
)
from fairseq2.datasets.preference import (
    GENERIC_PREFERENCE_DATASET_FAMILY,
    GenericPreferenceDataset,
)
from fairseq2.datasets.speech import GENERIC_SPEECH_DATASET_FAMILY, GenericSpeechDataset
from fairseq2.datasets.text import GENERIC_TEXT_DATASET_FAMILY, GenericTextDataset


def _register_datasets(context: RuntimeContext) -> None:
    file_system = context.file_system

    asset_download_manager = context.asset_download_manager

    registry = context.get_registry(DatasetHandler)

    handler: DatasetHandler

    # ASR
    handler = StandardDatasetHandler(
        GENERIC_ASR_DATASET_FAMILY,
        GenericAsrDataset,
        GenericAsrDataset.from_path,
        file_system,
        asset_download_manager,
    )

    registry.register(handler.family, handler)

    # Instruction
    handler = StandardDatasetHandler(
        GENERIC_INSTRUCTION_DATASET_FAMILY,
        GenericInstructionDataset,
        GenericInstructionDataset.from_path,
        file_system,
        asset_download_manager,
    )

    registry.register(handler.family, handler)

    # Parallel Text
    handler = StandardDatasetHandler(
        GENERIC_PARALLEL_TEXT_DATASET_FAMILY,
        GenericParallelTextDataset,
        GenericParallelTextDataset.from_path,
        file_system,
        asset_download_manager,
    )

    registry.register(handler.family, handler)

    # Preference Optimization
    handler = StandardDatasetHandler(
        GENERIC_PREFERENCE_DATASET_FAMILY,
        GenericPreferenceDataset,
        GenericPreferenceDataset.from_path,
        file_system,
        asset_download_manager,
    )

    registry.register(handler.family, handler)

    # Speech
    handler = StandardDatasetHandler(
        GENERIC_SPEECH_DATASET_FAMILY,
        GenericSpeechDataset,
        GenericSpeechDataset.from_path,
        file_system,
        asset_download_manager,
    )

    registry.register(handler.family, handler)

    # Text
    handler = StandardDatasetHandler(
        GENERIC_TEXT_DATASET_FAMILY,
        GenericTextDataset,
        GenericTextDataset.from_path,
        file_system,
        asset_download_manager,
    )

    registry.register(handler.family, handler)
