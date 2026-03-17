# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import SequenceBatch, SyncMode
from fairseq2.metrics import MetricBag
from fairseq2.recipe import EvalUnit, RecipeModel, Trainer, TrainUnit
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.error import RecipeError
from fairseq2.runtime.dependency import DependencyContainer

from .config import Wav2Vec2SslRecipeConfig
from .criterion import Wav2Vec2SslCriterion
from .data import (
    WAV2VEC2_SSL_DATASET,
    Wav2Vec2SslDataset,
    Wav2Vec2SslDatasetConfig,
    open_wav2vec2_ssl_dataset,
)


@final
class Wav2Vec2SslRecipe(TrainRecipe):
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
    def create_trainer(self, context: RecipeContext) -> Trainer:
        config = context.config.as_(Wav2Vec2SslRecipeConfig)

        criterion = Wav2Vec2SslCriterion(
            context.model,
            config.loss.diversity_loss_weight,
            config.loss.feature_penalty_weight,
        )

        unit = Wav2Vec2SslTrainUnit(criterion)
        dataset = context.default_dataset.as_(Wav2Vec2SslDataset)

        if config.dataset.train_split is None:
            raise RecipeError(
                "Wav2Vec2SslDatasetConfig.train_split must be defined for training but is `None`."
            )

        seed = config.common.seed

        data_reader = dataset.create_reader(
            config.dataset.train_split,  # type: ignore
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
            example_shuffle_window=config.dataset.example_shuffle_window,
            batch_shuffle_window=config.dataset.batch_shuffle_window,
            num_accumulate=config.trainer.grad_accumulation.num_batches,
            num_prefetch=config.dataset.num_prefetch,
            drop_remainder=config.dataset.drop_remainder,
            sync_mode=SyncMode.UNTIL_FIRST,
            seed=seed,
            max_num_batches=config.dataset.max_num_batches,
            cached_fd_count=config.dataset.cached_fd_count,
        )
        seed += 1

        valid_unit = Wav2Vec2SslEvalUnit(criterion)

        if config.dataset.valid_split is None:
            raise RecipeError(
                "Wav2Vec2SslDatasetConfig.valid_split must be defined for training but is `None`."
            )

        # Same parameters as training but with validation-specific settings
        valid_data_reader = dataset.create_reader(
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
            example_shuffle_window=1,  # No sample shuffling
            batch_shuffle_window=1,  # No batch shuffling
            num_accumulate=1,  # No grad accumulation
            num_prefetch=config.dataset.num_prefetch,
            drop_remainder=config.dataset.drop_remainder,
            sync_mode=SyncMode.UNTIL_LAST,  # Wait for all processes
            seed=seed,
            max_num_batches=config.dataset.max_num_batches,
            cached_fd_count=config.dataset.cached_fd_count,
        )

        return context.create_trainer(
            unit, data_reader, [valid_unit], [valid_data_reader]
        )

    @property
    @override
    def config_kls(self) -> type[object]:
        return Wav2Vec2SslRecipeConfig


@final
class Wav2Vec2SslTrainUnit(TrainUnit[SequenceBatch]):
    """wav2vec2 training unit"""

    _criterion: Wav2Vec2SslCriterion

    def __init__(self, criterion: Wav2Vec2SslCriterion) -> None:
        self._criterion = criterion

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        self._criterion.prepare_metric_bag(metric_bag)

    @override
    def process_batch(
        self, batch: SequenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        return self._criterion(batch, metric_bag)

    @property
    @override
    def model(self) -> RecipeModel:
        return self._criterion.model


@final
class Wav2Vec2SslEvalUnit(EvalUnit[SequenceBatch]):
    """wav2vec2 evaluation unit for validation during training."""

    _criterion: Wav2Vec2SslCriterion

    def __init__(self, criterion: Wav2Vec2SslCriterion) -> None:
        self._criterion = criterion

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        self._criterion.prepare_metric_bag(metric_bag)

    @override
    def process_batch(self, batch: SequenceBatch, metric_bag: MetricBag) -> None:
        self._criterion(batch, metric_bag)

    @property
    @override
    def model(self) -> RecipeModel:
        return self._criterion.model
