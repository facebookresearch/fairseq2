# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import SyncMode
from fairseq2.recipe import EvalRecipe, Evaluator, RecipeContext
from fairseq2.recipe.error import RecipeError
from fairseq2.runtime.dependency import DependencyContainer

from ..criterion import Wav2Vec2AsrCriterion
from ..data import (
    WAV2VEC2_ASR_DATASET,
    Wav2Vec2AsrDataset,
    Wav2Vec2AsrDatasetConfig,
    open_wav2vec2_asr_dataset,
)
from ..recipe import Wav2Vec2AsrEvalUnit
from ..wer_calculator import WerCalculator
from .config import Wav2Vec2AsrEvalRecipeConfig


@final
class Wav2Vec2AsrEvalRecipe(EvalRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            WAV2VEC2_ASR_DATASET,
            Wav2Vec2AsrDataset,
            Wav2Vec2AsrDatasetConfig,
            opener=open_wav2vec2_asr_dataset,
        )

    @override
    def create_evaluator(self, context: RecipeContext) -> Evaluator:
        config = context.config.as_(Wav2Vec2AsrEvalRecipeConfig)

        valid_criterion = Wav2Vec2AsrCriterion(
            model=context.model, wer_calculator=WerCalculator.from_context(context)
        )

        dataset = context.default_dataset.as_(Wav2Vec2AsrDataset)

        if config.dataset.valid_split is None:
            raise RecipeError(
                "`dataset.valid_split` must be defined for evaluation but is `None`."
            )

        seed = config.common.seed

        eval_unit = Wav2Vec2AsrEvalUnit(valid_criterion)

        eval_data_reader = dataset.create_reader(
            config.dataset.valid_split,
            context.default_tokenizer,
            context.gangs,
            min_audio_len=config.dataset.min_audio_len,
            max_audio_len=config.dataset.max_audio_len,
            # Batching parameters
            num_seqs_multiple_of=config.dataset.num_seqs_multiple_of,
            max_num_elements=config.dataset.max_num_elements,
            # Audio processing parameters
            dtype=config.dataset.dtype,
            normalize_audio=config.dataset.normalize_audio,
            npc=config.dataset.npc,
            # Shuffling and performance parameters
            example_shuffle_window=1,  # No pre-batch shuffling
            batch_shuffle_window=1,  # No batch shuffling
            num_accumulate=1,  # No accumulation
            num_prefetch=config.dataset.num_prefetch,
            drop_remainder=config.dataset.drop_remainder,
            sync_mode=SyncMode.UNTIL_LAST,
            seed=seed,
            max_num_batches=config.dataset.max_num_batches,
            cached_fd_count=config.dataset.cached_fd_count,
        )

        return context.create_evaluator([eval_unit], [eval_data_reader])

    @property
    @override
    def config_kls(self) -> type[object]:
        return Wav2Vec2AsrEvalRecipeConfig
