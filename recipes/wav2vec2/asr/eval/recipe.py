# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import final

from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import MetricBag
from fairseq2.metrics.text import WerMetric
from fairseq2.recipe import EvalRecipe, Evaluator, EvalUnit, RecipeContext, RecipeModel
from fairseq2.recipe.error import RecipeError
from fairseq2.runtime.dependency import DependencyContainer

from ..criterion import Wav2Vec2AsrCriterion
from ..data import (
    WAV2VEC2_ASR_DATASET,
    Wav2Vec2AsrDataset,
    Wav2Vec2AsrDatasetConfig,
    open_wav2vec2_asr_dataset,
)
from ..metrics import add_asr_metrics
from ..wer_calculator import WerCalculator
from .default_config import Wav2Vec2AsrEvalRecipeConfig


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

        # Evaluation equivalent of the training criterion
        valid_criterion = Wav2Vec2AsrCriterion(
            model=context.model, wer_calculator=WerCalculator.from_context(context)
        )

        dataset = context.default_dataset.as_(Wav2Vec2AsrDataset)

        if config.dataset.valid_split is None:
            raise RecipeError(
                "`dataset.valid_split` must be defined for evaluation but is `None`."
            )

        seed = config.common.seed

        # Initialize validation units
        eval_units = []
        eval_data_readers = []
        eval_splits = config.dataset.valid_split.split(",")

        for split in eval_splits:
            seed += 1

            eval_unit = Wav2Vec2AsrEvalUnit(valid_criterion)
            eval_units.append(eval_unit)

            eval_data_reader = dataset.create_reader(
                split,
                context.default_tokenizer,
                context.gangs,
                min_audio_len=config.dataset.min_audio_len,
                max_audio_len=config.dataset.max_audio_len,
                # Batching parameters
                batching_strategy=config.dataset.batching_strategy,
                batch_size=config.dataset.batch_size,
                num_seqs_multiple_of=config.dataset.num_seqs_multiple_of,
                max_num_elements=config.dataset.max_num_elements,
                # Audio processing parameters
                dtype=config.dataset.dtype,
                normalize_audio=config.dataset.normalize_audio,
                no_padding=config.dataset.no_padding,
                npc=config.dataset.npc,
                # Shuffling and performance parameters
                example_shuffle_window=config.dataset.example_shuffle_window,
                batch_shuffle_window=config.dataset.batch_shuffle_window,
                num_accumulate=1,
                num_prefetch=config.dataset.num_prefetch,
                drop_remainder=config.dataset.drop_remainder,
                sync_batches=config.dataset.sync_batches,
                sync_mode=config.dataset.sync_mode,
                seed=seed,
                max_num_batches=config.dataset.max_num_batches,
                cached_fd_count=config.dataset.cached_fd_count,
            )
            eval_data_readers.append(eval_data_reader)


        return context.create_evaluator(eval_units, eval_data_readers)

    @property
    @override
    def config_kls(self) -> type[object]:
        return Wav2Vec2AsrEvalRecipeConfig


@final
class Wav2Vec2AsrEvalUnit(EvalUnit[Seq2SeqBatch]):
    """
    wav2vec2 ASR evaluation unit.
    """

    _criterion: Wav2Vec2AsrCriterion

    def __init__(self, scorer: Wav2Vec2AsrCriterion) -> None:
        self._criterion = scorer

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        add_asr_metrics(metric_bag)

        metric_bag.add("wer", WerMetric())

    @override
    def process_batch(self, batch: Seq2SeqBatch, metric_bag: MetricBag) -> None:
        self._criterion(batch, metric_bag)

    @override
    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        """Accessor function to modify the logged values"""
        return self._criterion.process_metric_values(values)

    @property
    @override
    def model(self) -> RecipeModel:
        return self._criterion.model
