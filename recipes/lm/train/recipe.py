# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from typing_extensions import override

from fairseq2.datasets import SequenceBatch, register_dataset_family
from fairseq2.metrics import MetricBag
from fairseq2.metrics.common import update_nll_loss, update_seq_batch_metrics
from fairseq2.model import Model
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.trainer import Trainer, TrainUnit

from .config import LMTrainConfig
from .dataset import (
    LM_TRAIN_DATASET,
    LMTrainDataset,
    LMTrainDatasetConfig,
    open_lm_train_dataset,
)


@final
class LMTrainRecipe(TrainRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            LM_TRAIN_DATASET,
            LMTrainDataset,
            LMTrainDatasetConfig,
            opener=open_lm_train_dataset,
        )

    @override
    def create_trainer(self, context: RecipeContext) -> Trainer:
        config = context.config_as(LMTrainConfig)

        unit = LMTrainUnit(context.model)

        dataset = context.dataset_as(LMTrainDataset)

        data_reader = dataset.create_reader(
            context.tokenizer,
            context.gangs,
            max_seq_len=config.dataset.max_seq_len,
            max_num_tokens=config.dataset.max_num_tokens,
            num_accumulate=config.trainer.grad_accumulation.num_batches,
            prefetch=config.dataset.prefetch,
            sync_ranks=config.dataset.sync_ranks,
        )

        return context.create_trainer(unit, data_reader)

    @property
    @override
    def config_kls(self) -> type[object]:
        return LMTrainConfig


@final
class LMTrainUnit(TrainUnit[SequenceBatch]):
    def __init__(self, model: Model) -> None:
        self._model = model

    @override
    def __call__(
        self, batch: SequenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, None]:
        input_batch, target_batch = batch.as_auto_regressive()

        seqs, seqs_layout = input_batch.as_input()

        nll_loss = self._model.module(
            seqs, seqs_layout, targets=target_batch.seqs, reduction="mean"
        )

        update_nll_loss(metric_bag, nll_loss)

        update_seq_batch_metrics(metric_bag, batch)

        return nll_loss, None

    @property
    @override
    def model(self) -> Model:
        return self._model
