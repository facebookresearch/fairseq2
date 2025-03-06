# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, cast, final

import torch
import torch.distributed
from torch import Tensor
from torcheval.metrics import Mean
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.logging import log
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModelOutput,
    as_auto_regressive_input,
)
from fairseq2.nn.utils.module import freeze_parameters
from fairseq2.recipes import Model, TrainUnit
from fairseq2.recipes.common import setup_reference_model
from fairseq2.recipes.config import (
    ReferenceModelSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipes.lm._preference_finetune._common import (
    POCriterionSection,
    POFinetuneMetricBag,
    _gather_lprobs_avg,
)
from fairseq2.recipes.lm._preference_finetune._handler import POFinetuneUnitHandler
from fairseq2.typing import DataType
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@final
class DpoFinetuneUnit(TrainUnit[PreferenceBatch]):
    """Represents the language model DPO-finetuning unit. Paper: https://arxiv.org/abs/2305.18290."""

    _model: Model
    _reference_model: Model | None
    _beta: float
    _nll_scale: float
    _metric_bag: DpoFinetuneMetricBag
    _length_normalization: bool

    def __init__(
        self,
        model: Model,
        reference_model: Model | None,
        gangs: Gangs,
        beta: float = 0.1,
        nll_scale: float = 1.0,
        length_normalization: bool = False,
    ) -> None:
        self._model = model
        self._reference_model = reference_model
        self._beta = beta
        self._nll_scale = nll_scale
        self._length_normalization = length_normalization

        self._metric_bag = DpoFinetuneMetricBag(gangs.dp)

    @override
    def __call__(self, batch: PreferenceBatch) -> tuple[Tensor, int]:
        chosen_batch = batch.chosen
        chosen_input_batch, chosen_target_batch = as_auto_regressive_input(chosen_batch)
        rejected_batch = batch.rejected
        rejected_input_batch, rejected_target_batch = as_auto_regressive_input(
            rejected_batch
        )
        if (
            chosen_target_batch.target_mask is None
            or rejected_target_batch.target_mask is None
        ):
            raise RuntimeError("target_mask attributes must exist for DPO loss")

        chosen_output = cast(
            SequenceModelOutput, self._model.module(chosen_input_batch)
        )
        rejected_output = cast(
            SequenceModelOutput, self._model.module(rejected_input_batch)
        )

        chosen_logps, average_chosen_logps = _gather_lprobs_avg(
            chosen_output, chosen_target_batch
        )
        rejected_logps, average_rejected_logps = _gather_lprobs_avg(
            rejected_output, rejected_target_batch
        )

        if self._reference_model is not None:
            with torch.no_grad():
                ref_chosen_output = cast(
                    SequenceModelOutput, self._reference_model.module(chosen_batch)
                )
                ref_rejected_output = cast(
                    SequenceModelOutput, self._reference_model.module(rejected_batch)
                )
                ref_chosen_logps, ref_average_chosen_logps = _gather_lprobs_avg(
                    ref_chosen_output, chosen_target_batch
                )
                ref_rejected_logps, ref_average_rejected_logps = _gather_lprobs_avg(
                    ref_rejected_output, rejected_target_batch
                )
        elif (
            batch.reference_score_chosen is not None
            and batch.reference_score_rejected is not None
        ):
            # reference scores must exist in the batch if reference model is None
            ref_chosen_logps = batch.reference_score_chosen
            ref_average_chosen_logps = (
                ref_chosen_logps / chosen_target_batch.target_mask.sum(-1)
            )
            ref_rejected_logps = batch.reference_score_rejected
            ref_average_rejected_logps = (
                ref_rejected_logps / rejected_target_batch.target_mask.sum(-1)
            )
        else:
            raise RuntimeError(
                "Reference model is not initialized and data batch does not provide reference score, but at least one must exist."
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
        assert target.target_mask is not None
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

    @property
    @override
    def model(self) -> Model:
        return self._model

    @property
    @override
    def metric_bag(self) -> DpoFinetuneMetricBag:
        return self._metric_bag


class DpoFinetuneMetricBag(POFinetuneMetricBag):
    """Holds the metrics of a DPO preference finetuning task."""

    dpo_loss: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("dpo_loss", Mean(device=gang.device), persistent=False)

    @torch.inference_mode()
    def update_dpo_loss(self, batch: PreferenceBatch, loss: Tensor) -> None:
        """Update the DPO loss metric.

        :param batch:
            The batch processed by the model.
        :param loss:
            The DPO loss of ``batch``.
        """
        self.dpo_loss.update(
            loss / batch.chosen.batch_size, weight=batch.chosen.batch_size
        )


DPO_FINETUNE_UNIT: Final = "dpo"


@dataclass(kw_only=True)
class DpoFinetuneConfig:
    reference_model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="llama3_1_8b_instruct")
    )
    """
    The reference model. If ``None``, the recipe expects to get reference
    log-probabilities for chosen and rejected targets as float values in the
    data example (fields `reference_score_rejected` and  `reference_score_chosen`).
    """

    reference_dtype: DataType = torch.bfloat16
    """The data type of the reference model."""

    # Loss
    beta: float = 0.1
    """The coefficient of regularization towards the reference model."""

    nll_scale: float = 0.0
    """The coefficient of NLL loss added to the DPO loss."""

    length_normalization: bool = False
    """Use length normalized DPO, which uses the average log probability of a sequence as the implicit reward."""


@final
class DpoFinetuneUnitHandler(POFinetuneUnitHandler):
    _context: RuntimeContext

    def __init__(self, context: RuntimeContext) -> None:
        self._context = context

    @override
    def create(
        self, model: Model, gangs: Gangs, recipe_config: object
    ) -> TrainUnit[PreferenceBatch]:
        criterion_section = get_config_section(
            recipe_config, "criterion", POCriterionSection
        )

        config = structure(criterion_section.config, DpoFinetuneConfig)

        validate(config)

        if config.reference_model is not None:
            log.info("Setting up DPO with reference model.")

            trainer_section = get_config_section(
                recipe_config, "trainer", TrainerSection
            )

            reference_model = setup_reference_model(
                DecoderModel,
                self._context,
                config.reference_model.name,
                gangs,
                config.reference_dtype,
                mp=False,
                torch_compile=trainer_section.torch_compile,
            )

            freeze_parameters(reference_model.module)

            log.info("DPO setup complete.")
        else:
            reference_model = None

        return DpoFinetuneUnit(
            model,
            reference_model,
            gangs,
            config.beta,
            config.nll_scale,
            config.length_normalization,
        )

    @property
    @override
    def name(self) -> str:
        return DPO_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return DpoFinetuneConfig
