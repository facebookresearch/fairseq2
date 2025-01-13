# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.datasets import DatasetHandler, DatasetLoader, StandardDatasetHandler
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
    GENERIC_PREFERENCE_OPTIMIZATION_DATASET_FAMILY,
    GenericPreferenceOptimizationDataset,
)
from fairseq2.datasets.speech import GENERIC_SPEECH_DATASET_FAMILY, GenericSpeechDataset
from fairseq2.datasets.text import GENERIC_TEXT_DATASET_FAMILY, GenericTextDataset


def _register_datasets(context: RuntimeContext) -> None:
    register_dataset(
        context,
        GENERIC_ASR_DATASET_FAMILY,
        kls=GenericAsrDataset,
        loader=GenericAsrDataset.from_path,
    )

    register_dataset(
        context,
        GENERIC_INSTRUCTION_DATASET_FAMILY,
        kls=GenericInstructionDataset,
        loader=GenericInstructionDataset.from_path,
    )

    register_dataset(
        context,
        GENERIC_PARALLEL_TEXT_DATASET_FAMILY,
        kls=GenericParallelTextDataset,
        loader=GenericParallelTextDataset.from_path,
    )

    register_dataset(
        context,
        GENERIC_PREFERENCE_OPTIMIZATION_DATASET_FAMILY,
        kls=GenericPreferenceOptimizationDataset,
        loader=GenericPreferenceOptimizationDataset.from_path,
    )

    register_dataset(
        context,
        GENERIC_SPEECH_DATASET_FAMILY,
        kls=GenericSpeechDataset,
        loader=GenericSpeechDataset.from_path,
    )

    register_dataset(
        context,
        GENERIC_TEXT_DATASET_FAMILY,
        kls=GenericTextDataset,
        loader=GenericTextDataset.from_path,
    )


def register_dataset(
    context: RuntimeContext, family: str, *, kls: type, loader: DatasetLoader
) -> None:
    handler = StandardDatasetHandler(kls, loader, context.asset_download_manager)

    registry = context.get_registry(DatasetHandler)

    registry.register(family, handler)
