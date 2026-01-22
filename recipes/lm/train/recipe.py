# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import SequenceBatch
from fairseq2.metrics import MetricBag
from fairseq2.metrics.common import (
    add_nll_loss_metric,
    add_seq_batch_metrics,
    update_nll_loss_metric,
    update_seq_batch_metrics,
)
from fairseq2.models.clm import CausalLM
from fairseq2.recipe.base import Recipe, RecipeContext
from fairseq2.recipe.trainer import TrainUnit
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.task import Task

from ..common import check_model_vocabulary
from .config import LMTrainConfig
from .dataset import (
    LM_TRAIN_DATASET,
    LMTrainDataset,
    LMTrainDatasetConfig,
    open_lm_train_dataset,
)


class LMTrainRecipe(Recipe):
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
    def create_task(self, context: RecipeContext) -> Task:
        config = context.get_config_as(LMTrainConfig)

        check_model_vocabulary(context)

        dp_model = context.get_data_parallel_model()

        unit = LMTrainUnit(dp_model)

        dataset = context.get_dataset_as(LMTrainDataset)

        tokenizer = context.get_tokenizer()

        data_reader = dataset.create_reader(
            tokenizer,
            context.gangs,
            max_seq_len=config.dataset.max_seq_len,
            max_num_tokens=config.dataset.max_num_tokens,
            num_accumulate=config.trainer.grad_accumulation.num_batches,
            seed=config.common.seed,
            prefetch=config.dataset.prefetch,
            sync_ranks=config.dataset.sync_ranks,
        )

        return context.create_trainer(unit, data_reader)

    @property
    @override
    def config_kls(self) -> type[object]:
        return LMTrainConfig


class LMTrainUnit(TrainUnit[SequenceBatch]):
    def __init__(self, model: Module) -> None:
        self._model = model

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        add_nll_loss_metric(metric_bag)

        add_seq_batch_metrics(metric_bag)

    @override
    def process_batch(
        self, batch: SequenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, None]:
        model = cast(CausalLM, self._model)

        input_batch, target_batch = batch.as_auto_regressive()

        seqs, seqs_layout = input_batch.as_input()

        nll_loss = model(seqs, seqs_layout, targets=target_batch.seqs, reduction="mean")

        update_nll_loss_metric(metric_bag, nll_loss)

        update_seq_batch_metrics(metric_bag, batch)

        return nll_loss, None
