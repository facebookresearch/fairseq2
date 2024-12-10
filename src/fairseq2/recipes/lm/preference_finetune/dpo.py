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

# from fairseq2.recipes.lm.preference_finetune.recipe import preference_unit_factory
from fairseq2.recipes.lm.preference_finetune.utils import (
    PreferenceFinetuneMetricBag,
    _gather_lprobs_avg,
    _load_reference_model,
    preference_unit_factory,
)
from fairseq2.recipes.trainer import AbstractTrainUnit
from fairseq2.recipes.utils.asset import AssetReference
from fairseq2.typing import DataType

log = get_log_writer(__name__)


@final
class DpoFinetuneUnit(AbstractTrainUnit[PreferenceOptimizationBatch]):
    """Represents the language model DPO-finetuning unit. Paper: https://arxiv.org/abs/2305.18290."""

    _reference_model: Module
    _beta: float
    _nll_scale: float
    _metric_bag: DpoFinetuneMetricBag
    _length_normalization: bool

    def __init__(
        self,
        model: Module,
        reference_model: Module,
        gang: Gang,
        beta: float = 0.1,
        nll_scale: float = 1.0,
        length_normalization: bool = False,
    ) -> None:
        super().__init__(model)

        self._reference_model = reference_model
        self._beta = beta
        self._nll_scale = nll_scale
        self._length_normalization = length_normalization

        self._metric_bag = DpoFinetuneMetricBag(gang)

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

        with torch.no_grad():
            ref_chosen_output = cast(
                SequenceModelOutput, self._reference_model(chosen_batch)
            )
            ref_rejected_output = cast(
                SequenceModelOutput, self._reference_model(rejected_batch)
            )
            ref_chosen_logps, ref_average_chosen_logps = _gather_lprobs_avg(
                ref_chosen_output, chosen_target_batch
            )
            ref_rejected_logps, ref_average_rejected_logps = _gather_lprobs_avg(
                ref_rejected_output, rejected_target_batch
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

        nll_loss = chosen_output.compute_loss(
            chosen_target_batch.seqs, loss_mask=chosen_target_batch.target_mask
        )

        self._metric_bag.update_dpo_loss(batch, dpo_loss)

        self._metric_bag.update_nll_loss(chosen_batch, nll_loss)

        self._metric_bag.update_sequence_lengths(batch)

        self._metric_bag.update_logps(batch, chosen_logps, rejected_logps)

        self._metric_bag.update_batch_metrics(chosen_batch)

        loss = (
            dpo_loss
            + self._nll_scale
            * nll_loss
            * chosen_target_batch.batch_size
            / chosen_target_batch.num_target_elements()
        )  # normalization applied locally per-rank

        return loss, chosen_target_batch.batch_size

    def _gather_lprobs(
        self, output: SequenceModelOutput, target: SequenceBatch
    ) -> tuple[Tensor, Tensor]:
        logprobs = torch.log_softmax(output.logits, dim=-1)
        per_token_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(
            -1
        )
        total_logps = (per_token_logps * target.target_mask).sum(dim=-1)  # [Batch, 1]
        assert target.target_mask is not None
        average_logps = total_logps / target.target_mask.sum(-1)

        return total_logps, average_logps

    def _compute_dpo_loss(
        self,
        chosen_logps: Tensor,
        ref_chosen_logps: Tensor,
        rejected_logps: Tensor,
        ref_rejected_logps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logp_ratio_chosen = self._beta * (chosen_logps - ref_chosen_logps)
        logp_ratio_rejected = self._beta * (rejected_logps - ref_rejected_logps)
        dpo_loss = -torch.nn.functional.logsigmoid(
            logp_ratio_chosen - logp_ratio_rejected
        )
        return logp_ratio_chosen, logp_ratio_rejected, dpo_loss.sum()

    @override
    def set_step_nr(self, step_nr: int) -> None:
        self._step_nr = step_nr

    @property
    @override
    def metric_bag(self) -> DpoFinetuneMetricBag:
        return self._metric_bag


register_metric_formatter("dpo_loss", "DPO Loss", 0, format_as_float)


class DpoFinetuneMetricBag(PreferenceFinetuneMetricBag):
    """Holds the metrics of a DPO preference finetuning task."""

    dpo_loss: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("dpo_loss", Mean(device=gang.device), persistent=False)

    @torch.inference_mode()
    def update_dpo_loss(self, batch: PreferenceOptimizationBatch, loss: Tensor) -> None:
        """Update the DPO loss metric.

        :param batch:
            The batch processed by the model.
        :param loss:
            The DPO loss of ``batch``.
        """
        self.dpo_loss.update(
            loss / batch.chosen.batch_size, weight=batch.chosen.batch_size
        )


@dataclass(kw_only=True)
class DpoConfig:
    """Holds the DPO configuration of a language model preference-finetuning task."""

    # Reference Model
    reference_model: AssetReference = "llama3_1_8b_instruct"
    """The name, path, or path to the asset card of the reference model."""

    reference_dtype: DataType = torch.bfloat16
    """The data type of the reference model."""

    reference_tensor_parallel_size: int = 1
    """The size of tensor parallelism for the reference model."""

    # Loss
    beta: float = 0.1
    """The coefficient of regularization towards the reference model."""

    nll_scale: float = 0.0
    """The coefficient of NLL loss added to the DPO loss."""

    length_normalization: bool = False
    """Use length normalized DPO, which uses the average log probability of a sequence as the implicit reward."""


@preference_unit_factory("dpo")
def create_dpo_unit(
    config: DpoConfig, model: Module, root_gang: Gang, gangs: Mapping[str, Gang]
) -> DpoFinetuneUnit:
    reference_model = _load_reference_model(
        config.reference_model,
        config.reference_dtype,
        root_gang,
        gangs,
        config.reference_tensor_parallel_size,
        log,
    )

    dp_gang = gangs["dp"]  # data

    return DpoFinetuneUnit(
        model,
        reference_model,
        dp_gang,
        config.beta,
        config.nll_scale,
        config.length_normalization,
    )
