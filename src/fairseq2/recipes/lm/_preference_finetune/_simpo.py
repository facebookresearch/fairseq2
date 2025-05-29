# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, final

import torch
import torch.distributed
from torch import Tensor
from typing_extensions import override

from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.device import Device
from fairseq2.gang import Gangs
from fairseq2.metrics import Mean, MetricBag
from fairseq2.recipes import Model, TrainUnit
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

# isort: split

from fairseq2.recipes.lm._preference_finetune._common import (
    POFinetuneMetricBag,
    _gather_lprobs_avg,
)
from fairseq2.recipes.lm._preference_finetune._config import POFinetuneConfig
from fairseq2.recipes.lm._preference_finetune._handler import POFinetuneUnitHandler


@final
class SimPOFinetuneUnit(TrainUnit[PreferenceBatch]):
    """Represents the language model SimPO-finetuning unit. Paper: https://arxiv.org/abs/2405.14734."""

    _model: Model
    _beta: float
    _gamma: float
    _nll_scale: float
    _metric_bag: SimPOFinetuneMetricBag

    def __init__(
        self,
        model: Model,
        beta: float = 0.1,
        gamma: float = 0.5,
        nll_scale: float = 1.0,
    ) -> None:
        self._model = model
        self._beta = beta
        self._gamma = gamma
        self._nll_scale = nll_scale

        self._metric_bag = SimPOFinetuneMetricBag(device=model.device)

    @override
    def __call__(self, batch: PreferenceBatch) -> tuple[Tensor, int]:
        chosen_batch = batch.chosen
        chosen_input_batch, chosen_target_batch = chosen_batch.as_auto_regressive()

        rejected_batch = batch.rejected
        rejected_input_batch, rejected_target_batch = (
            rejected_batch.as_auto_regressive()
        )

        chosen_seqs, chosen_seqs_layout = chosen_input_batch.as_input()

        nll_loss, chosen_logits = self._model.module(
            chosen_seqs,
            chosen_seqs_layout,
            targets=chosen_target_batch.seqs,
            target_mask=chosen_target_batch.target_mask,
            return_logits=True,
        )

        rejected_seqs, rejected_seqs_layout = rejected_input_batch.as_input()

        rejected_logits = self._model.module(rejected_seqs, rejected_seqs_layout)

        chosen_logps, average_chosen_logps = _gather_lprobs_avg(
            chosen_logits, chosen_target_batch
        )
        rejected_logps, average_rejected_logps = _gather_lprobs_avg(
            rejected_logits, rejected_target_batch
        )

        simpo_loss = self._compute_simpo_loss(
            average_chosen_logps, average_rejected_logps
        )

        self._metric_bag.update_simpo_loss(batch, simpo_loss)

        self._metric_bag.update_nll_loss(chosen_batch, nll_loss)

        self._metric_bag.update_sequence_lengths(batch)

        self._metric_bag.update_logps(batch, chosen_logps, rejected_logps)

        self._metric_bag.update_batch_metrics(chosen_batch)

        loss = (
            simpo_loss
            + self._nll_scale
            * nll_loss
            * chosen_target_batch.batch_size
            / chosen_target_batch.num_target_elements
        )  # nll normalization applied locally per-rank

        return loss, chosen_target_batch.batch_size

    def _compute_simpo_loss(
        self, average_chosen_logps: Tensor, average_rejected_logps: Tensor
    ) -> Tensor:
        simpo_loss = -torch.nn.functional.logsigmoid(
            self._beta * (average_chosen_logps - average_rejected_logps) - self._gamma
        )
        return simpo_loss.sum()

    @property
    @override
    def model(self) -> Model:
        return self._model

    @property
    @override
    def metric_bag(self) -> MetricBag:
        return self._metric_bag


class SimPOFinetuneMetricBag(POFinetuneMetricBag):
    """Holds the metrics of a SimPO preference finetuning task."""

    simpo_loss: Mean

    def __init__(self, device: Device) -> None:
        super().__init__(device)

        self.simpo_loss = Mean(device=device)

    @torch.inference_mode()
    def update_simpo_loss(self, batch: PreferenceBatch, loss: Tensor) -> None:
        self.simpo_loss.update(
            loss / batch.chosen.batch_size, weight=batch.chosen.batch_size
        )


SIMPO_FINETUNE_UNIT: Final = "simpo"


@dataclass(kw_only=True)
class SimPOFinetuneConfig:
    beta: float = 1
    """The coefficient of KL-divergence regularization."""

    gamma: float = 0.5
    """The target reward margin between positive and negative completions."""

    nll_scale: float = 0.0
    """The coefficient of NLL loss added to the SimPO loss."""


@final
class SimPOFinetuneUnitHandler(POFinetuneUnitHandler):
    @override
    def create(
        self, model: Model, gangs: Gangs, recipe_config: POFinetuneConfig
    ) -> TrainUnit[PreferenceBatch]:
        config = structure(recipe_config.criterion.config, SimPOFinetuneConfig)

        validate(config)

        return SimPOFinetuneUnit(model, config.beta, config.gamma, config.nll_scale)

    @property
    @override
    def name(self) -> str:
        return SIMPO_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return SimPOFinetuneConfig
