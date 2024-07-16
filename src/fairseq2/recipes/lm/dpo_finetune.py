from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union, cast, final

import torch
import torch.distributed
from torch import Tensor
from torch.nn import Module
from torcheval.metrics import Mean

from fairseq2.config_registry import ConfigRegistry
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
from fairseq2.recipes.lm.preference_finetune import PreferenceOptimizationConfig
from fairseq2.recipes.trainer import AbstractTrainUnit
from fairseq2.typing import override

log = get_log_writer(__name__)


@dataclass
class DpoFinetuneConfig(PreferenceOptimizationConfig):
    """Holds the DPO-finetuning configuration of a language model."""

    dpo_beta: float = 0.1

    nll_scale: float = 1.0


dpo_finetune_presets = ConfigRegistry[DpoFinetuneConfig]()

dpo_finetune_preset = dpo_finetune_presets.decorator


@dpo_finetune_preset("llama3_8b_instruct")
def _llama3_8b_instruct() -> DpoFinetuneConfig:
    cfg = DpoFinetuneConfig()
    cfg.max_num_tokens = 1000
    cfg.max_seq_len = 1000
    cfg.max_gradient_norm = 1.0
    return cfg


# batch size and min lengths are tuned for OA2 in this preset!
@dpo_finetune_preset("llama3_70b_instruct_openassistant2")
def _llama3_70b_instruct_openassistant2() -> DpoFinetuneConfig:
    cfg = DpoFinetuneConfig()
    cfg.model = "llama3_70b_instruct"
    cfg.reference_model = "llama3_70b_instruct"
    cfg.tensor_parallel_size = 8
    cfg.max_num_tokens = (
        200  # 70B DPO training might catch OOM, tune the effective batch size if needed
    )
    cfg.max_seq_len = 200
    cfg.max_gradient_norm = 1.0
    cfg.gradient_accumulation = 8  # to address small batch size
    return cfg


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
        reference_model: Union[Module | None],
        gang: Gang,
        beta=0.1,
        nll_scale=1.0,
    ) -> None:
        super().__init__(model)

        assert (
            reference_model is not None
        )  # is this the best way to asset this? Is there an error already built for this?
        self._reference_model = reference_model
        self._beta = beta
        self._nll_scale = nll_scale

        self._metric_bag = DpoFinetuneMetricBag(gang)
        self._gang = gang

    @override
    def __call__(self, batch: PreferenceOptimizationBatch) -> Tuple[Tensor, int]:
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
            f"Step:{self._step_nr} Rank:{get_rank()} IDs:{','.join(batch.chosen.example['id'])}, DPO loss: {dpo_loss.item()}"
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
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
