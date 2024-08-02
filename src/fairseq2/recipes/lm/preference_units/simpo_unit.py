from __future__ import annotations

from dataclasses import dataclass
from typing import cast, final

import torch
import torch.distributed
from torch import Tensor
from torch.nn import Module
from torcheval.metrics import Mean
from typing_extensions import override

from fairseq2.datasets.preference import PreferenceOptimizationBatch
from fairseq2.gang import Gang, get_rank
from fairseq2.logging import get_log_writer
from fairseq2.metrics.recorder import format_as_float, register_metric_formatter
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModelOutput,
    as_auto_regressive_input,
)
from fairseq2.recipes.common_metrics import SequenceMetricBag
from fairseq2.recipes.trainer import AbstractTrainUnit

log = get_log_writer(__name__)


@dataclass
class SimpoFinetuneConfig:
    """Holds the SimPO-finetuning configuration of a language model."""

    # Hyperparameters
    simpo_beta: float = 1
    """The coefficient of KL-divergence regularization."""

    simpo_gamma: float = 0.5
    """Target reward margin between positive and negative completions."""

    nll_scale: float = 0.0
    """The coefficient of NLL loss added to the SimPO loss."""


@final
class SimpoFinetuneUnit(AbstractTrainUnit[PreferenceOptimizationBatch]):
    """Represents the DPO-finetuning unit of a language model."""

    _beta: float
    _gamma: float
    _nll_scale: float
    _metric_bag: SimpoFinetuneMetricBag

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

        self._metric_bag = SimpoFinetuneMetricBag(gang)
        self._gang = gang

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

        chosen_logps, average_chosen_logps = self._gather_lprobs(
            chosen_output, chosen_target_batch
        )
        rejected_logps, average_rejected_logps = self._gather_lprobs(
            rejected_output, rejected_target_batch
        )

        simpo_loss = self._compute_simpo_loss(
            average_chosen_logps, average_rejected_logps
        )

        nll_loss = chosen_output.compute_loss(
            chosen_target_batch.seqs, loss_mask=chosen_target_batch.target_mask
        )

        # adding NLL loss to the total loss for now!
        loss = simpo_loss + self._nll_scale * nll_loss

        log.info(
            f"Step:{self._step_nr} Rank:{get_rank()} IDs:{[str(idx) for idx in batch.chosen.example['id']]}, SimPO loss: {simpo_loss.item()}"
        )

        self._metric_bag.update_nll_loss(chosen_batch, nll_loss)
        self._metric_bag.update_simpo_loss(chosen_batch, simpo_loss)

        self._metric_bag.update_batch_metrics(chosen_batch)

        return loss, chosen_target_batch.batch_size

    def _gather_lprobs(
        self, output: SequenceModelOutput, target: SequenceBatch
    ) -> tuple[Tensor, Tensor]:
        logprobs = torch.log_softmax(output.logits, dim=-1)
        per_token_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(
            -1
        )
        total_logps = (per_token_logps * target.target_mask).sum(dim=-1)  # [Batch, 1]
        assert (
            target.target_mask is not None
        )  # TODO hacky mypy fix - perhaps use the length of the per_token_logps?
        average_logps = total_logps / target.target_mask.sum(-1)

        return total_logps, average_logps

    def _compute_simpo_loss(
        self,
        average_chosen_logps: Tensor,
        average_rejected_logps: Tensor,
    ) -> Tensor:
        simpo_loss = -torch.nn.functional.logsigmoid(
            self._beta * (average_chosen_logps - average_rejected_logps) - self._gamma
        )
        return simpo_loss.sum()

    @property
    @override
    def metric_bag(self) -> SimpoFinetuneMetricBag:
        return self._metric_bag

    def set_step_nr(self, step_nr: int) -> None:
        """Set the current training step number."""
        self._step_nr = step_nr


register_metric_formatter("simpo_loss", "SimPO Loss", 0, format_as_float)


class SimpoFinetuneMetricBag(SequenceMetricBag):
    _simpo_loss: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("_simpo_loss", Mean(device=gang.device), persistent=False)

    @torch.inference_mode()
    def update_simpo_loss(self, batch: SequenceBatch, loss: Tensor) -> None:
        batch_size = torch.tensor(batch.batch_size)

        normalized_loss = loss.cpu() / batch_size

        self._simpo_loss.update(normalized_loss, weight=batch_size)
