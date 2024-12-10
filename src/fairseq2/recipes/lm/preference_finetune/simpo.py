# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast, final

import torch
import torch.distributed
from torch import Tensor
from torch.nn import Module
from torcheval.metrics import Mean
from typing_extensions import override

from fairseq2.datasets.preference import PreferenceOptimizationBatch
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics.recorder import format_as_float, register_metric_formatter
from fairseq2.models.sequence import SequenceModelOutput, as_auto_regressive_input
from fairseq2.recipes.lm.preference_finetune.utils import (
    PreferenceFinetuneMetricBag,
    _gather_lprobs_avg,
    preference_unit_factory,
)
from fairseq2.recipes.trainer import AbstractTrainUnit

log = get_log_writer(__name__)


@final
class SimPOFinetuneUnit(AbstractTrainUnit[PreferenceOptimizationBatch]):
    """Represents the language model SimPO-finetuning unit. Paper: https://arxiv.org/abs/2405.14734."""

    _beta: float
    _gamma: float
    _nll_scale: float
    _metric_bag: SimPOFinetuneMetricBag

    def __init__(
        self,
        model: Module,
        gang: Gang,
        beta: float = 0.1,
        gamma: float = 0.5,
        nll_scale: float = 1.0,
    ) -> None:
        super().__init__(model)

        self._beta = beta
        self._gamma = gamma
        self._nll_scale = nll_scale

        self._metric_bag = SimPOFinetuneMetricBag(gang)

    @override
    def __call__(self, batch: PreferenceOptimizationBatch) -> tuple[Tensor, int]:
        chosen_batch = batch.chosen
        chosen_input_batch, chosen_target_batch = as_auto_regressive_input(chosen_batch)
        rejected_batch = batch.rejected
        rejected_input_batch, rejected_target_batch = as_auto_regressive_input(
            rejected_batch
        )

        chosen_output = cast(SequenceModelOutput, self._model(chosen_input_batch))
        rejected_output = cast(SequenceModelOutput, self._model(rejected_input_batch))

        chosen_logps, average_chosen_logps = _gather_lprobs_avg(
            chosen_output, chosen_target_batch
        )
        rejected_logps, average_rejected_logps = _gather_lprobs_avg(
            rejected_output, rejected_target_batch
        )

        simpo_loss = self._compute_simpo_loss(
            average_chosen_logps, average_rejected_logps
        )

        nll_loss = chosen_output.compute_loss(
            chosen_target_batch.seqs, loss_mask=chosen_target_batch.target_mask
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
            / chosen_target_batch.num_target_elements()
        )  # nll normalization applied locally per-rank

        return loss, chosen_target_batch.batch_size

    def _compute_simpo_loss(
        self, average_chosen_logps: Tensor, average_rejected_logps: Tensor
    ) -> Tensor:
        simpo_loss = -torch.nn.functional.logsigmoid(
            self._beta * (average_chosen_logps - average_rejected_logps) - self._gamma
        )
        return simpo_loss.sum()

    @override
    def set_step_nr(self, step_nr: int) -> None:
        self._step_nr = step_nr

    @property
    @override
    def metric_bag(self) -> SimPOFinetuneMetricBag:
        return self._metric_bag


register_metric_formatter("simpo_loss", "SimPO Loss", 0, format_as_float)


class SimPOFinetuneMetricBag(PreferenceFinetuneMetricBag):
    """Holds the metrics of a SimPO preference finetuning task."""

    simpo_loss: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("simpo_loss", Mean(device=gang.device), persistent=False)

    @torch.inference_mode()
    def update_simpo_loss(
        self, batch: PreferenceOptimizationBatch, loss: Tensor
    ) -> None:
        """Update the SimPO loss metric.

        :param batch:
            The batch processed by the model.
        :param loss:
            The SimPO loss of ``batch``.
        """
        self.simpo_loss.update(
            loss / batch.chosen.batch_size, weight=batch.chosen.batch_size
        )


@dataclass(kw_only=True)
class SimPOConfig:
    """Holds the SimPO configuration of a language model preference-finetuning task."""

    beta: float = 1
    """The coefficient of KL-divergence regularization."""

    gamma: float = 0.5
    """The target reward margin between positive and negative completions."""

    nll_scale: float = 0.0
    """The coefficient of NLL loss added to the SimPO loss."""


@preference_unit_factory("simpo")
def create_simpo_unit(
    config: SimPOConfig, model: Module, root_gang: Gang, gangs: Mapping[str, Gang]
) -> SimPOFinetuneUnit:
    dp_gang = gangs["dp"]  # data

    return SimPOFinetuneUnit(
        model, dp_gang, config.beta, config.gamma, config.nll_scale
    )
