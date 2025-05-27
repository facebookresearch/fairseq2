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
    _gather_lprobs,
)
from fairseq2.recipes.lm._preference_finetune._config import POFinetuneConfig
from fairseq2.recipes.lm._preference_finetune._handler import POFinetuneUnitHandler


@final
class OrpoFinetuneUnit(TrainUnit[PreferenceBatch]):
    """Represents the language model ORPO-finetuning unit. Paper: https://arxiv.org/abs/2403.07691."""

    _model: Model
    _lambda: float
    _nll_scale: float
    _metric_bag: OrpoFinetuneMetricBag

    def __init__(
        self,
        model: Model,
        orpo_lambda: float = 1.0,
        nll_scale: float = 1.0,
    ) -> None:
        self._model = model
        self._lambda = orpo_lambda
        self._nll_scale = nll_scale

        self._metric_bag = OrpoFinetuneMetricBag(device=model.device)

    @override
    def __call__(self, batch: PreferenceBatch) -> tuple[Tensor, int]:
        chosen_batch = batch.chosen
        chosen_input_batch, chosen_target_batch = chosen_batch.as_auto_regressive()

        rejected_batch = batch.rejected
        rejected_input_batch, rejected_target_batch = (
            rejected_batch.as_auto_regressive()
        )

        chosen_seqs, chosen_seqs_layout = chosen_input_batch.as_input()

        nll_loss, chosen_logits = self._model(
            chosen_seqs,
            chosen_seqs_layout,
            targets=chosen_target_batch.seqs,
            target_mask=chosen_target_batch.target_mask,
            return_logits=True,
        )

        rejected_seqs, rejected_seqs_layout = rejected_input_batch.as_input()

        rejected_logits = self._model(rejected_seqs, rejected_seqs_layout)

        chosen_logps = _gather_lprobs(chosen_logits, chosen_target_batch)
        rejected_logps = _gather_lprobs(rejected_logits, rejected_target_batch)

        orpo_loss = self._compute_orpo_loss(chosen_logps, rejected_logps)

        self._metric_bag.update_orpo_loss(batch, orpo_loss)

        self._metric_bag.update_nll_loss(chosen_batch, nll_loss)

        self._metric_bag.update_sequence_lengths(batch)

        self._metric_bag.update_logps(batch, chosen_logps, rejected_logps)

        self._metric_bag.update_batch_metrics(chosen_batch)

        loss = (
            orpo_loss
            + self._nll_scale
            * nll_loss
            * chosen_target_batch.batch_size
            / chosen_target_batch.num_target_elements
        )  # normalization applied locally per-rank

        return loss, chosen_target_batch.batch_size

    def _compute_orpo_loss(
        self,
        chosen_logps: Tensor,
        rejected_logps: Tensor,
    ) -> Tensor:
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps))
            - torch.log1p(-torch.exp(rejected_logps))
        )

        orpo_loss = -torch.nn.functional.logsigmoid(log_odds)
        return orpo_loss.sum()

    @property
    @override
    def model(self) -> Model:
        return self._model

    @property
    @override
    def metric_bag(self) -> MetricBag:
        return self._metric_bag


class OrpoFinetuneMetricBag(POFinetuneMetricBag):
    """Holds the metrics of a ORPO preference finetuning task."""

    orpo_loss: Mean

    def __init__(self, device: Device) -> None:
        super().__init__(device)

        self.orpo_loss = Mean(device=device)

    @torch.inference_mode()
    def update_orpo_loss(self, batch: PreferenceBatch, loss: Tensor) -> None:
        self.orpo_loss.update(
            loss / batch.chosen.batch_size, weight=batch.chosen.batch_size
        )


ORPO_FINETUNE_UNIT: Final = "orpo"


@dataclass(kw_only=True)
class OrpoFinetuneConfig:
    orpo_lambda: float = 1.0
    """The coefficient of the odds-ratio component of ORPO loss"""

    nll_scale: float = 1.0
    """The coefficient of the NLL component of ORPO loss."""


@final
class OrpoFinetuneUnitHandler(POFinetuneUnitHandler):
    @override
    def create(
        self, model: Model, gangs: Gangs, recipe_config: POFinetuneConfig
    ) -> TrainUnit[PreferenceBatch]:
        config = structure(recipe_config.criterion.config, OrpoFinetuneConfig)

        validate(config)

        return OrpoFinetuneUnit(model, config.orpo_lambda, config.nll_scale)

    @property
    @override
    def name(self) -> str:
        return ORPO_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return OrpoFinetuneConfig
