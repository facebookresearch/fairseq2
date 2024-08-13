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
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModelOutput,
    as_auto_regressive_input,
)
from fairseq2.recipes.common_metrics import SequenceMetricBag
from fairseq2.recipes.lm.preference_finetune.recipe import preference_unit_factory
from fairseq2.recipes.trainer import AbstractTrainUnit

log = get_log_writer(__name__)


@final
class CPOFinetuneUnit(AbstractTrainUnit[PreferenceOptimizationBatch]):
    """Represents the language model CPO-finetuning unit. Paper: https://arxiv.org/abs/2401.08417."""

    _beta: float
    _nll_scale: float
    _metric_bag: CPOFinetuneMetricBag

    def __init__(
        self,
        model: Module,
        gang: Gang,
        beta: float = 1.0,
        nll_scale: float = 1.0,
    ) -> None:
        super().__init__(model)

        self._beta = beta
        self._nll_scale = nll_scale

        self._metric_bag = CPOFinetuneMetricBag(gang)

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

        chosen_logps = self._gather_lprobs(chosen_output, chosen_target_batch)
        rejected_logps = self._gather_lprobs(rejected_output, rejected_target_batch)

        cpo_loss = self._compute_cpo_loss(chosen_logps, rejected_logps)

        nll_loss = chosen_output.compute_loss(
            chosen_target_batch.seqs, loss_mask=chosen_target_batch.target_mask
        )

        # adding NLL loss to the total loss for now!
        loss = cpo_loss + self._nll_scale * nll_loss

        self._metric_bag.update_nll_loss(chosen_batch, nll_loss)

        self._metric_bag.update_cpo_loss(chosen_batch, cpo_loss)

        self._metric_bag.update_batch_metrics(chosen_batch)

        return loss, chosen_target_batch.batch_size

    def _gather_lprobs(
        self, output: SequenceModelOutput, target: SequenceBatch
    ) -> Tensor:
        logprobs = torch.log_softmax(output.logits, dim=-1)
        chosen_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(-1)
        chosen_logps = (chosen_logps * target.target_mask).sum(dim=-1)  # [Batch, 1]

        return chosen_logps

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
    def metric_bag(self) -> CPOFinetuneMetricBag:
        return self._metric_bag


register_metric_formatter("cpo_loss", "CPO Loss", 0, format_as_float)


class CPOFinetuneMetricBag(SequenceMetricBag):
    _cpo_loss: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("_cpo_loss", Mean(device=gang.device), persistent=False)

    @torch.inference_mode()
    def update_cpo_loss(self, batch: SequenceBatch, loss: Tensor) -> None:
        self._cpo_loss.update(loss / batch.batch_size, weight=batch.batch_size)


@dataclass(kw_only=True)
class CPOConfig:
    """Holds the CPO configuration of a language model preference-finetuning task."""

    # Hyperparameters
    beta: float = 1.0
    """The coefficient applied to the difference between preferred and dispreferred sequences."""

    nll_scale: float = 1.0
    """The coefficient of NLL loss added to the CPO loss."""


@preference_unit_factory("cpo")
def create_cpo_unit(
    config: CPOConfig, model: Module, root_gang: Gang, gangs: Mapping[str, Gang]
) -> CPOFinetuneUnit:
    dp_gang = gangs["dp"]  # data

    return CPOFinetuneUnit(model, dp_gang, config.beta, config.nll_scale)
