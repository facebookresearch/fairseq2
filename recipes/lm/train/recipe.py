# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import cast, final

from torch import Tensor
from typing_extensions import override

from fairseq2.datasets import LengthBatching, SequenceBatch, register_dataset_family
from fairseq2.dependency import DependencyContainer
from fairseq2.metrics import MetricBag
from fairseq2.models.clm import CausalLM
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.metrics import update_nll_loss, update_seq_batch_metrics
from fairseq2.recipe.model import Model
from fairseq2.recipe.run import train
from fairseq2.recipe.trainer import Trainer, TrainUnit

from .config import CausalLMTrainConfig
from .dataset import (
    JSONL_TEXT_DATASET_FAMILY,
    JsonlTextDataset,
    TextReadOptions,
)


@final
class CausalLMTrainRecipe(TrainRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            JSONL_TEXT_DATASET_FAMILY,
            JsonlTextDataset,
            JsonlTextDataset.from_path,
        )

    @override
    def load_trainer(self, context: RecipeContext) -> Trainer:
        config = context.config_as(CausalLMTrainConfig)

        unit = CausalLMTrainUnit(context.model)

        gangs = context.gangs

        dataset = context.resolver.resolve(object, key="dataset")

        dataset = cast(JsonlTextDataset, dataset)

        tokenizer = context.tokenizer

        text_encoder = tokenizer.create_encoder(mode="default")

        seed = context.next_seed()

        batching = LengthBatching(config.dataset.max_num_tokens)

        read_options = TextReadOptions(
            batching=batching,
            num_accumulate=config.trainer.grad_accumulation.num_batches,
            num_prefetch=config.dataset.num_prefetch,
            seed=seed,
            extras=config.dataset.extras,
        )

        data_reader = dataset.create_reader(
            text_encoder,
            tokenizer.vocab_info.pad_idx,
            gangs.dp,
            config.dataset.min_seq_len,
            config.dataset.max_seq_len,
            read_options,
        )

        return context.create_trainer(unit, data_reader)

    @property
    @override
    def config_kls(self) -> type[object]:
        return CausalLMTrainConfig


@final
class CausalLMTrainUnit(TrainUnit[SequenceBatch]):
    _model: Model

    def __init__(self, model: Model) -> None:
        self._model = model

    @override
    def __call__(
        self, batch: SequenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, None]:
        batch, target_batch = batch.as_auto_regressive()

        seqs, seqs_layout = batch.as_input()

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


def train_clm(config: CausalLMTrainConfig, output_dir: Path) -> None:
    recipe = CausalLMTrainRecipe()

    train(recipe, config, output_dir)
