# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast, final

import torch
import torch.distributed
from torch import Tensor
from typing_extensions import override

from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.metrics import Mean
from fairseq2.models.sequence import SequenceModelOutput, as_auto_regressive_input
from fairseq2.recipes import Model, TrainUnit
from fairseq2.recipes.config import get_config_section
from fairseq2.recipes.lm._preference_finetune._common import (
    POCriterionSection,
    POFinetuneMetricBag,
    _gather_lprobs,
)
from fairseq2.recipes.lm._preference_finetune._handler import POFinetuneUnitHandler
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


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
        gangs: Gangs,
        orpo_lambda: float = 1.0,
        nll_scale: float = 1.0,
    ) -> None:
        self._model = model
        self._lambda = orpo_lambda
        self._nll_scale = nll_scale

        self._metric_bag = OrpoFinetuneMetricBag(gangs.dp)

    @override
    def __call__(self, batch: PreferenceBatch) -> tuple[Tensor, int]:
        chosen_batch = batch.chosen
        chosen_input_batch, chosen_target_batch = as_auto_regressive_input(chosen_batch)
        rejected_batch = batch.rejected
        rejected_input_batch, rejected_target_batch = as_auto_regressive_input(
            rejected_batch
        )

        chosen_output = cast(
            SequenceModelOutput, self._model.module(chosen_input_batch)
        )
        rejected_output = cast(
            SequenceModelOutput, self._model.module(rejected_input_batch)
        )

        chosen_logps = _gather_lprobs(chosen_output, chosen_target_batch)
        rejected_logps = _gather_lprobs(rejected_output, rejected_target_batch)

        orpo_loss = self._compute_orpo_loss(chosen_logps, rejected_logps)

        nll_loss = chosen_output.compute_loss(
            chosen_target_batch.seqs, loss_mask=chosen_target_batch.target_mask
        )

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
            / chosen_target_batch.num_target_elements()
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
    def metric_bag(self) -> OrpoFinetuneMetricBag:
        return self._metric_bag


class OrpoFinetuneMetricBag(POFinetuneMetricBag):
    """Holds the metrics of a ORPO preference finetuning task."""

    orpo_loss: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("orpo_loss", Mean(device=gang.device), persistent=False)

    @torch.inference_mode()
    def update_orpo_loss(self, batch: PreferenceBatch, loss: Tensor) -> None:
        """Update the ORPO loss metric.

        :param batch:
            The batch processed by the model.
        :param loss:
            The ORPO loss of ``batch``.
        """
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
        self, model: Model, gangs: Gangs, recipe_config: object
    ) -> TrainUnit[PreferenceBatch]:
        criterion_section = get_config_section(
            recipe_config, "criterion", POCriterionSection
        )

        config = structure(criterion_section.config, OrpoFinetuneConfig)

        validate(config)

        return OrpoFinetuneUnit(model, gangs, config.orpo_lambda, config.nll_scale)

    @property
    @override
    def name(self) -> str:
        return ORPO_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return OrpoFinetuneConfig
