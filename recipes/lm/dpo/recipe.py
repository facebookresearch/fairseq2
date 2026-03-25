# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import SequenceBatch
from fairseq2.metrics import MetricBag
from fairseq2.metrics.common import (
    add_nll_loss_metric,
    add_seq_batch_metrics,
    add_dpo_loss_metric,
    add_sequence_length_metrics,
    add_logps_metrics,
    update_nll_loss_metric,
    update_seq_batch_metrics,
    update_dpo_loss_metric,
    update_sequence_length_metrics,
    update_logps_metrics,
)
from fairseq2.models.clm import CausalLM
from fairseq2.recipe.base import Recipe, RecipeContext
from fairseq2.recipe.trainer import TrainUnit
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.task import Task

from ..common import check_model_vocabulary, Batching, LengthBatching, StaticBatching, _gather_lprobs_avg
from .config import LMDPOConfig
from .dataset import (
    LM_DPO_DATASET,
    LMDPODataReadOptions,
    LMDPODataset,
    LMDPODatasetConfig,
    open_lm_dpo_dataset,
    PreferenceBatch,
)


class LMDPORecipe(Recipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            LM_DPO_DATASET,
            LMDPODataset,
            LMDPODatasetConfig,
            opener=open_lm_dpo_dataset,
        )

    @override
    def create_task(self, context: RecipeContext) -> Task:
        config = context.get_config_as(LMDPOConfig)

        check_model_vocabulary(context)

        dp_model = context.get_data_parallel_model()

        # Get reference model if configured
        reference_model = None
        if config.reference_model is not None:
            reference_model = context.bootstrap_reference_model("reference_model")

        unit = LMDPOUnit(
            dp_model,
            reference_model,
            beta=config.beta,
            nll_scale=config.nll_scale,
            length_normalization=config.length_normalization,
        )

        dataset = context.get_dataset_as(LMDPODataset)

        tokenizer = context.get_tokenizer()

        batching: Batching
        if config.dataset.batch_size is not None:
            batching = StaticBatching(config.dataset.batch_size)
        else:
            batching = LengthBatching(config.dataset.max_num_tokens)

        read_options = LMDPODataReadOptions(
            batching=batching,
            example_shuffle_window=config.dataset.example_shuffle_window,
            batch_shuffle_window=config.dataset.batch_shuffle_window,
            num_accumulate=config.trainer.grad_accumulation.num_batches,
            prefetch=config.dataset.prefetch,
            source_encode_mode=config.dataset.source_encode_mode,
            target_encode_mode=config.dataset.target_encode_mode,
            chat_mode=config.dataset.chat_mode,
            seed=config.common.seed,
        )

        data_reader = dataset.create_reader(
            tokenizer=tokenizer,
            gangs=context.gangs,
            min_seq_len=config.dataset.min_seq_len,
            max_seq_len=config.dataset.max_seq_len,
            options=read_options,
        )

        return context.create_trainer(unit, data_reader, [], [])

    @property
    @override
    def config_kls(self) -> type[object]:
        return LMDPOConfig


class LMDPOUnit(TrainUnit[PreferenceBatch]):
    def __init__(
        self,
        model: Module,
        reference_model: Module | None,
        beta: float = 0.1,
        nll_scale: float = 1.0,
        length_normalization: bool = False,
    ) -> None:
        self._model = model
        self._reference_model = reference_model
        self._beta = beta
        self._nll_scale = nll_scale
        self._length_normalization = length_normalization

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:  # TODO: add metrics
        add_nll_loss_metric(metric_bag)

        add_seq_batch_metrics(metric_bag)

        add_dpo_loss_metric(metric_bag)

        add_sequence_length_metrics(metric_bag)

        add_logps_metrics(metric_bag)

    def _compute_dpo_loss(
        self,
        chosen_logps: Tensor,
        ref_chosen_logps: Tensor,
        rejected_logps: Tensor,
        ref_rejected_logps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logp_ratio_chosen = self._beta * (chosen_logps - ref_chosen_logps)
        logp_ratio_rejected = self._beta * (rejected_logps - ref_rejected_logps)
        dpo_loss = - Module.functional.logsigmoid(
            logp_ratio_chosen - logp_ratio_rejected
        )
        return logp_ratio_chosen, logp_ratio_rejected, dpo_loss.sum()

    @override
    def process_batch(  # TODO: update metrics
        self, batch: PreferenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        model = cast(CausalLM, self._model)
        reference_model = cast(CausalLM, self._reference_model) if self._reference_model else None

        chosen_batch = batch.chosen
        chosen_input_batch, chosen_target_batch = chosen_batch.as_auto_regressive()

        rejected_batch = batch.rejected
        rejected_input_batch, rejected_target_batch = (
            rejected_batch.as_auto_regressive()
        )

        if (
            chosen_target_batch.target_mask is None
            or rejected_target_batch.target_mask is None
        ):
            raise RuntimeError("target_mask attributes must exist for DPO loss")

        chosen_seqs, chosen_seqs_layout = chosen_input_batch.as_input()

        nll_loss, chosen_logits = model(
            chosen_seqs,
            chosen_seqs_layout,
            targets=chosen_target_batch.seqs,
            target_mask=chosen_target_batch.target_mask,
            return_logits=True,
        )

        rejected_seqs, rejected_seqs_layout = rejected_input_batch.as_input()

        rejected_logits = model(rejected_seqs, rejected_seqs_layout)

        chosen_logps, average_chosen_logps = _gather_lprobs_avg(
            chosen_logits, chosen_target_batch
        )
        rejected_logps, average_rejected_logps = _gather_lprobs_avg(
            rejected_logits, rejected_target_batch
        )

        if reference_model is not None:
            chosen_seqs, chosen_seqs_layout = chosen_batch.as_input()
            rejected_seqs, rejected_seqs_layout = rejected_batch.as_input()

            with torch.no_grad():
                ref_chosen_logits = reference_model(
                    chosen_seqs, chosen_seqs_layout
                )
                ref_rejected_logits = reference_model(
                    rejected_seqs, rejected_seqs_layout
                )

                ref_chosen_logps, ref_average_chosen_logps = _gather_lprobs_avg(
                    ref_chosen_logits, chosen_target_batch
                )
                ref_rejected_logps, ref_average_rejected_logps = _gather_lprobs_avg(
                    ref_rejected_logits, rejected_target_batch
                )
        elif (
            batch.reference_score_chosen is not None
            and batch.reference_score_rejected is not None
        ):
            # reference scores must exist in the batch if reference model is None
            ref_chosen_logps = batch.reference_score_chosen
            ref_average_chosen_logps = (
                ref_chosen_logps / chosen_target_batch.target_mask.sum(-1)
            )
            ref_rejected_logps = batch.reference_score_rejected
            ref_average_rejected_logps = (
                ref_rejected_logps / rejected_target_batch.target_mask.sum(-1)
            )
        else:
            raise RuntimeError(
                "Reference model is not initialized and data batch does not provide reference score, but at least one must exist."
            )

        if self._length_normalization:
            _, _, dpo_loss = self._compute_dpo_loss(
                average_chosen_logps,
                ref_average_chosen_logps,
                average_rejected_logps,
                ref_average_rejected_logps,
            )
        else:
            _, _, dpo_loss = self._compute_dpo_loss(
                chosen_logps, ref_chosen_logps, rejected_logps, ref_rejected_logps
            )

        update_dpo_loss_metric(metric_bag, dpo_loss, batch)

        update_nll_loss_metric(metric_bag, nll_loss, chosen_batch.num_target_elements)

        update_sequence_length_metrics(metric_bag, batch)

        update_logps_metrics(metric_bag, batch, chosen_logps, rejected_logps)

        update_seq_batch_metrics(metric_bag, chosen_batch)
        
        loss = (
            dpo_loss
            + self._nll_scale
            * nll_loss
            * chosen_target_batch.batch_size
            / chosen_target_batch.num_target_elements
        )  # normalization applied locally per-rank

        return loss, chosen_target_batch.batch_size
