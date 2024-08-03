from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
from fairseq2.typing import DataType

log = get_log_writer(__name__)


@dataclass(kw_only=True)
class DpoFinetuneConfig:
    """Holds the DPO-finetuning configuration of a language model."""

    # Hyperparameters
    dpo_beta: float = 0.1
    """The coefficient of regularization towards the reference model."""

    nll_scale: float = 0.0
    """The coefficient of NLL loss added to the DPO loss."""

    # Reference Model
    reference_model: str | Path = "llama3_8b_instruct"
    """The name or path to the asset card of the reference model to use."""

    reference_dtype: DataType = torch.bfloat16
    """The data type of the reference model."""

    reference_tensor_parallel_size: int = 1
    """The size of tensor parallelism for the reference model."""


@final
class DpoFinetuneUnit(AbstractTrainUnit[PreferenceOptimizationBatch]):
    """Represents the DPO-finetuning unit of a language model."""

    _reference_model: Module
    _beta: float
    _nll_scale: float
    _metric_bag: DpoFinetuneMetricBag

    def __init__(
        self,
        model: Module,
        reference_model: Module,
        gang: Gang,
        beta: float = 0.1,
        nll_scale: float = 1.0,
    ) -> None:
        super().__init__(model)

        self._reference_model = reference_model
        self._beta = beta
        self._nll_scale = nll_scale

        self._metric_bag = DpoFinetuneMetricBag(gang)
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

        chosen_logps = self._gather_lprobs(chosen_output, chosen_target_batch)
        rejected_logps = self._gather_lprobs(rejected_output, rejected_target_batch)

        with torch.no_grad():
            ref_chosen_output = cast(
                SequenceModelOutput, self._reference_model(chosen_batch)
            )
            ref_rejected_output = cast(
                SequenceModelOutput, self._reference_model(rejected_batch)
            )
            ref_chosen_logps = self._gather_lprobs(
                ref_chosen_output, chosen_target_batch
            )
            ref_rejected_logps = self._gather_lprobs(
                ref_rejected_output, rejected_target_batch
            )

        _, _, dpo_loss = self._compute_dpo_loss(
            chosen_logps, ref_chosen_logps, rejected_logps, ref_rejected_logps
        )

        nll_loss = chosen_output.compute_loss(
            chosen_target_batch.seqs, loss_mask=chosen_target_batch.target_mask
        )

        # adding NLL loss to the total loss for now!
        loss = dpo_loss + self._nll_scale * nll_loss

        log.info(
            f"Step:{self._step_nr} Rank:{get_rank()} IDs:{[str(idx) for idx in batch.chosen.example['id']]}, DPO loss: {dpo_loss.item()}"
        )

        self._metric_bag.update_nll_loss(chosen_batch, nll_loss)
        self._metric_bag.update_dpo_loss(chosen_batch, dpo_loss)

        self._metric_bag.update_batch_metrics(chosen_batch)

        return loss, chosen_target_batch.batch_size

    def _gather_lprobs(
        self, output: SequenceModelOutput, target: SequenceBatch
    ) -> Tensor:
        logprobs = torch.log_softmax(output.logits, dim=-1)
        chosen_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(-1)
        chosen_logps = (chosen_logps * target.target_mask).sum(dim=-1)  # [Batch, 1]

        return chosen_logps

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

    @property
    @override
    def metric_bag(self) -> DpoFinetuneMetricBag:
        return self._metric_bag

    def set_step_nr(self, step_nr: int) -> None:
        """Set the current training step number."""
        self._step_nr = step_nr


register_metric_formatter("dpo_loss", "DPO Loss", 0, format_as_float)


class DpoFinetuneMetricBag(SequenceMetricBag):
    _dpo_loss: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("_dpo_loss", Mean(device=gang.device), persistent=False)

    @torch.inference_mode()
    def update_dpo_loss(self, batch: SequenceBatch, loss: Tensor) -> None:
        batch_size = torch.tensor(batch.batch_size)

        normalized_loss = loss.cpu() / batch_size

        self._dpo_loss.update(normalized_loss, weight=batch_size)
