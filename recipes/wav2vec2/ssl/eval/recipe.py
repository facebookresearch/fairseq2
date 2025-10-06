# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from typing_extensions import override

from fairseq2.datasets import SequenceBatch, register_dataset_family
from fairseq2.evaluator import Evaluator, EvalUnit
from fairseq2.metrics import MetricBag
from fairseq2.recipe import RecipeModel
from fairseq2.recipe.base import EvalRecipe, RecipeContext
from fairseq2.runtime.dependency import DependencyContainer

from ..criterion import Wav2Vec2SslCriterion
from ..data import (
    WAV2VEC2_SSL_DATASET,
    Wav2Vec2SslDataset,
    Wav2Vec2SslDatasetConfig,
    open_wav2vec2_ssl_dataset,
)
from .default_config import Wav2Vec2SslEvalRecipeConfig


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
            raise ValueError(
                "Wav2Vec2SslDatasetConfig.valid_split must be defined for evaluation but is `None`."
            )

        seed = config.common.seed

        # Initialize validation units
        eval_units = []
        eval_data_readers = []
        eval_splits = config.dataset.valid_split.split(",")

        for split in eval_splits:
            seed += 1

            eval_unit = Wav2Vec2SslEvalUnit(criterion)
            eval_units.append(eval_unit)

            eval_data_reader = dataset.create_reader(
                split,
                context.gangs,
                min_audio_len=config.dataset.min_audio_len,
                max_audio_len=config.dataset.max_audio_len,
                # Batching parameters
                batching_strategy=config.dataset.batching_strategy,
                batch_size=config.dataset.batch_size,
                num_seqs_multiple_of=config.dataset.num_seqs_multiple_of,
                max_num_elements=config.dataset.max_num_elements,
                # Audio processing parameters
                dtype=config.evaluator.dtype,
                normalize_audio=config.dataset.normalize_audio,
                use_fbank=config.dataset.use_fbank,
                no_padding=config.dataset.no_padding,
                npc=config.dataset.npc,
                # SpecAugment parameters
                spec_aug_p=config.dataset.spec_aug_p,
                spec_aug_freq_mask_param=config.dataset.spec_aug_freq_mask_param,
                spec_aug_time_mask_param=config.dataset.spec_aug_time_mask_param,
                # Shuffling and performance parameters
                example_shuffle_window=config.dataset.example_shuffle_window,
                batch_shuffle_window=config.dataset.batch_shuffle_window,
                num_accumulate=config.trainer.grad_accumulation.num_batches,
                num_prefetch=config.dataset.num_prefetch,
                drop_remainder=config.dataset.drop_remainder,
                sync_batches=config.dataset.sync_batches,
                sync_mode=config.dataset.sync_mode,
                seed=context.next_seed(),
                max_num_batches=config.dataset.max_num_batches,
                cached_fd_count=config.dataset.cached_fd_count,
            )
            eval_data_readers.append(eval_data_reader)

        return context.create_evaluator(eval_units, eval_data_readers)

    @property
    @override
    def config_kls(self) -> type[object]:
        return Wav2Vec2SslEvalRecipeConfig


@final
class Wav2Vec2SslEvalUnit(EvalUnit[SequenceBatch]):
    """
    wav2vec2 SSL evaluation unit. Identical implementation to the similar
    named unit in the training recipe, but can be modified if needed.
    """

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
