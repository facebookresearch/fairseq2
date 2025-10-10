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
from fairseq2.datasets import SequenceBatch
from fairseq2.metrics import MetricBag
from fairseq2.metrics.common import (
    add_nll_loss_metric,
    add_seq_batch_metrics,
    update_nll_loss_metric,
    update_seq_batch_metrics,
)
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.evaluator import EvalUnit
from fairseq2.recipe.model import RecipeModel
from fairseq2.recipe.trainer import Trainer, TrainUnit
from fairseq2.runtime.dependency import DependencyContainer

from .config import LMSFTConfig
from .dataset import (
    LM_SFT_PADDED_DATASET,
    DataReadOptions,
    LMSFTDataset,
    LMSFTDatasetConfig,
    open_lm_sft_dataset,
)
from .utils import Batching, LengthBatching, StaticBatching


@final
class LMSFTRecipe(TrainRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            LM_SFT_PADDED_DATASET,
            LMSFTDataset,
            LMSFTDatasetConfig,
            opener=open_lm_sft_dataset,
        )

    @override
    def create_trainer(self, context: RecipeContext) -> Trainer:
        config = context.config.as_(LMSFTConfig)

        unit = LMSFTUnit(context.model)

        dataset = context.default_dataset.as_(LMSFTDataset)

        seed = config.common.seed

        seed += 1

        batching: Batching
        if config.dataset.batch_size is not None:
            batching = StaticBatching(config.dataset.batch_size)
        else:
            batching = LengthBatching(config.dataset.max_num_tokens)

        read_options = DataReadOptions(
            batching=batching,
            example_shuffle_window=config.dataset.example_shuffle_window,
            batch_shuffle_window=config.dataset.batch_shuffle_window,
            num_accumulate=config.trainer.grad_accumulation.num_batches,
            prefetch=config.dataset.prefetch,
            source_encode_mode=config.dataset.source_encode_mode,
            target_encode_mode=config.dataset.target_encode_mode,
            chat_mode=config.dataset.chat_mode,
            seed=seed,
            extras=config.dataset.extras,
        )

        data_reader = dataset.create_reader(
            config.dataset.train_split,
            context.default_tokenizer,
            context.gangs,
            min_seq_len=config.dataset.min_seq_len,
            max_seq_len=config.dataset.max_seq_len,
            options=read_options,
        )

        # Initialize the validation unit.
        if config.dataset.valid_split is not None:

            valid_unit = SFTLossEvalUnit(context.model)

            max_num_tokens = (
                config.dataset.max_num_valid_tokens or config.dataset.max_num_tokens
            )

            valid_batching: Batching = batching  # LengthBatching(max_num_tokens)

            read_options = DataReadOptions(
                batching=valid_batching,
                prefetch=config.dataset.prefetch,
                source_encode_mode=config.dataset.source_encode_mode,
                target_encode_mode=config.dataset.target_encode_mode,
                chat_mode=config.dataset.chat_mode,
                seed=seed,
                extras=config.dataset.extras,
            )

            valid_data_reader = dataset.create_reader(
                config.dataset.train_split,
                context.default_tokenizer,
                context.gangs,
                min_seq_len=config.dataset.min_seq_len,
                max_seq_len=config.dataset.max_seq_len,
                options=read_options,
            )

            valid_units = [valid_unit]
            valid_data_readers = [valid_data_reader]
        else:
            valid_units = []
            valid_data_readers = []

        seed += 1

        return context.create_trainer(
            unit=unit,
            data_reader=data_reader,
            valid_units=valid_units,
            valid_data_readers=valid_data_readers,
        )

    @property
    @override
    def config_kls(self) -> type[object]:
        return LMSFTConfig


@final
class LMSFTUnit(TrainUnit[SequenceBatch]):
    def __init__(self, model: RecipeModel) -> None:
        self._model = model

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        add_nll_loss_metric(metric_bag)
        add_seq_batch_metrics(metric_bag)

    @override
    def process_batch(
        self, batch: SequenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        input_batch, target_batch = batch.as_auto_regressive()

        seqs, seqs_layout = input_batch.as_input()

        nll_loss = self._model.module(
            seqs,
            seqs_layout,
            targets=target_batch.seqs,
            target_mask=target_batch.target_mask,
        )

        update_nll_loss_metric(
            metric_bag, nll_loss, num_targets=target_batch.num_target_elements
        )

        update_seq_batch_metrics(metric_bag, target_batch)

        return nll_loss, target_batch.num_target_elements

    @property
    @override
    def model(self) -> RecipeModel:
        return self._model


@final
class SFTLossEvalUnit(EvalUnit[SequenceBatch]):
    def __init__(self, model: RecipeModel) -> None:
        self._model = model

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        add_nll_loss_metric(metric_bag)
        add_seq_batch_metrics(metric_bag)

    @override
    def process_batch(self, batch: SequenceBatch, metric_bag: MetricBag) -> None:
        input_batch, target_batch = batch.as_auto_regressive()

        seqs, seqs_layout = input_batch.as_input()

        # breakpoint()

        nll_loss = self._model.module(
            seqs,
            seqs_layout,
            targets=target_batch.seqs,
            target_mask=target_batch.target_mask,
        )

        update_nll_loss_metric(
            metric_bag, nll_loss, num_targets=target_batch.num_target_elements
        )

        update_seq_batch_metrics(metric_bag, target_batch)

        return None

    @property
    @override
    def model(self) -> RecipeModel:
        return self._model
