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
from torch.nn import Module
from typing_extensions import override

from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.metrics import Mean
from fairseq2.models.sequence import SequenceModelOutput, as_auto_regressive_input
from fairseq2.recipes.config import get_config_section
from fairseq2.recipes.lm._preference_finetune._common import (
    POCriterionSection,
    POFinetuneMetricBag,
    _gather_lprobs,
)
from fairseq2.recipes.lm._preference_finetune._handler import POFinetuneUnitHandler
from fairseq2.recipes.trainer import AbstractTrainUnit, TrainUnit
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@final
class CpoFinetuneUnit(AbstractTrainUnit[PreferenceBatch]):
    """Represents the language model CPO-finetuning unit. Paper: https://arxiv.org/abs/2401.08417."""

    _beta: float
    _nll_scale: float
    _metric_bag: CpoFinetuneMetricBag

    def __init__(
        self,
        model: Module,
        gangs: Gangs,
        beta: float = 1.0,
        nll_scale: float = 1.0,
    ) -> None:
        super().__init__(model)

        self._beta = beta
        self._nll_scale = nll_scale

        self._metric_bag = CpoFinetuneMetricBag(gangs.dp)

    @override
    def __call__(self, batch: PreferenceBatch) -> tuple[Tensor, int]:
        chosen_batch = batch.chosen
        chosen_input_batch, chosen_target_batch = as_auto_regressive_input(chosen_batch)
        rejected_batch = batch.rejected
        rejected_input_batch, rejected_target_batch = as_auto_regressive_input(
            rejected_batch
        )

        chosen_output = cast(SequenceModelOutput, self._model(chosen_input_batch))
        rejected_output = cast(SequenceModelOutput, self._model(rejected_input_batch))

        chosen_logps = _gather_lprobs(chosen_output, chosen_target_batch)
        rejected_logps = _gather_lprobs(rejected_output, rejected_target_batch)

        cpo_loss = self._compute_cpo_loss(chosen_logps, rejected_logps)

        nll_loss = chosen_output.compute_loss(
            chosen_target_batch.seqs, loss_mask=chosen_target_batch.target_mask
        )

        self._metric_bag.update_cpo_loss(batch, cpo_loss)

        self._metric_bag.update_nll_loss(chosen_batch, nll_loss)

        self._metric_bag.update_sequence_lengths(batch)

        self._metric_bag.update_logps(batch, chosen_logps, rejected_logps)

        self._metric_bag.update_batch_metrics(chosen_batch)

        loss = (
            cpo_loss
            + self._nll_scale
            * nll_loss
            * chosen_target_batch.batch_size
            / chosen_target_batch.num_target_elements()
        )  # normalization applied locally per-rank

        return loss, chosen_target_batch.batch_size

    def _compute_cpo_loss(
        self,
        chosen_logps: Tensor,
        rejected_logps: Tensor,
    ) -> Tensor:
        cpo_loss = -torch.nn.functional.logsigmoid(
            self._beta * (chosen_logps - rejected_logps)
        )
        return cpo_loss.sum()

    @override
    def set_step_nr(self, step_nr: int) -> None:
        """Set the current training step number."""
        self._step_nr = step_nr

    @property
    @override
    def metric_bag(self) -> CpoFinetuneMetricBag:
        return self._metric_bag


class CpoFinetuneMetricBag(POFinetuneMetricBag):
    """Holds the metrics of a CPO preference finetuning task."""

    cpo_loss: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("cpo_loss", Mean(device=gang.device), persistent=False)

    @torch.inference_mode()
    def update_cpo_loss(self, batch: PreferenceBatch, loss: Tensor) -> None:
        """Update the CPO loss metric.

        :param batch:
            The batch processed by the model.
        :param loss:
            The CPO loss of ``batch``.
        """
        self.cpo_loss.update(
            loss / batch.chosen.batch_size, weight=batch.chosen.batch_size
        )


CPO_FINETUNE_UNIT: Final = "cpo"


@dataclass(kw_only=True)
class CpoFinetuneConfig:
    beta: float = 1.0
    """The coefficient applied to the difference between preferred and dispreferred sequences."""

    nll_scale: float = 1.0
    """The coefficient of NLL loss added to the CPO loss."""


@final
class CpoFinetuneUnitHandler(POFinetuneUnitHandler):
    @override
    def create(
        self, model: Module, gangs: Gangs, recipe_config: object
    ) -> TrainUnit[PreferenceBatch]:
        criterion_section = get_config_section(
            recipe_config, "criterion", POCriterionSection
        )

        config = structure(criterion_section.config, CpoFinetuneConfig)

        validate(config)

        return CpoFinetuneUnit(model, gangs, config.beta, config.nll_scale)

    @property
    @override
    def config_kls(self) -> type[object]:
        return CpoFinetuneConfig
