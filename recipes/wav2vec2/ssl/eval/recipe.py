# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.datasets.data_reader import SyncMode
from fairseq2.recipe import Evaluator
from fairseq2.recipe.base import EvalRecipe, RecipeContext
from fairseq2.recipe.error import RecipeError
from fairseq2.runtime.dependency import DependencyContainer

from ..criterion import Wav2Vec2SslCriterion
from ..data import (
    WAV2VEC2_SSL_DATASET,
    Wav2Vec2SslDataset,
    Wav2Vec2SslDatasetConfig,
    open_wav2vec2_ssl_dataset,
)
from ..recipe import Wav2Vec2SslEvalUnit
from .config import Wav2Vec2SslEvalRecipeConfig


@final
class Wav2Vec2SslEvalRecipe(EvalRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            WAV2VEC2_SSL_DATASET,
            Wav2Vec2SslDataset,
            Wav2Vec2SslDatasetConfig,
            opener=open_wav2vec2_ssl_dataset,
        )

    @override
    def create_evaluator(self, context: RecipeContext) -> Evaluator:
        config = context.config.as_(Wav2Vec2SslEvalRecipeConfig)

        criterion = Wav2Vec2SslCriterion(
            context.model,
            config.loss.diversity_loss_weight,
            config.loss.feature_penalty_weight,
        )

        dataset = context.default_dataset.as_(Wav2Vec2SslDataset)

        if config.dataset.valid_split is None:
            raise RecipeError(
                "Wav2Vec2SslDatasetConfig.valid_split must be defined for evaluation but is `None`."
            )

        seed = config.common.seed

        eval_unit = Wav2Vec2SslEvalUnit(criterion)

        eval_data_reader = dataset.create_reader(
            config.dataset.valid_split,
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
            example_shuffle_window=1,
            batch_shuffle_window=1,
            num_accumulate=1,
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
        return Wav2Vec2SslEvalRecipeConfig
