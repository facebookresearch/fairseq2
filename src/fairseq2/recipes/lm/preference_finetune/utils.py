# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
from torch.nn import Module
from torcheval.metrics import Mean

from fairseq2.datasets.preference import PreferenceOptimizationBatch
from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.gang import Gang
from fairseq2.logging import LogWriter
from fairseq2.metrics.recorder import format_as_float, register_metric_formatter
from fairseq2.models import load_model
from fairseq2.models.sequence import SequenceBatch, SequenceModelOutput
from fairseq2.nn.utils.module import freeze_parameters
from fairseq2.recipes.common_metrics import SequenceMetricBag
from fairseq2.recipes.trainer import TrainUnit
from fairseq2.recipes.utils.asset import AssetReference, retrieve_asset_card
from fairseq2.recipes.utils.setup import broadcast_model
from fairseq2.typing import META, DataType


def _load_reference_model(
    model_name_or_card: AssetReference,
    dtype: DataType,
    root_gang: Gang,
    gangs: Mapping[str, Gang],
    tensor_parallel_size: int,
    log: LogWriter,
) -> Module:
    dp_gang = gangs["dp"]

    card = retrieve_asset_card(model_name_or_card)

    log.info("Loading {} reference model on data parallel rank 0 (per shard).", card.name)  # fmt: skip

    if dp_gang.rank == 0:
        init_device = root_gang.device
    else:
        init_device = META

    # TODO: figure out how to load the reference model onto its own gangs
    model = load_model(card, gangs=gangs, device=init_device, dtype=dtype)

    root_gang.barrier()

    log.info("Reference model loaded on data parallel rank 0.")

    model.eval()

    freeze_parameters(model)

    # Distribute the model to all processes in the gang.
    if dp_gang.size != 1:
        broadcast_model(model, dp_gang, log)

    return model


def _gather_lprobs(output: SequenceModelOutput, target: SequenceBatch) -> Tensor:
    logprobs = torch.log_softmax(output.logits, dim=-1)
    chosen_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(-1)
    chosen_logps = (chosen_logps * target.target_mask).sum(dim=-1)  # [Batch, 1]

    return chosen_logps


def _gather_lprobs_avg(
    output: SequenceModelOutput, target: SequenceBatch
) -> tuple[Tensor, Tensor]:
    logprobs = torch.log_softmax(output.logits, dim=-1)
    per_token_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(-1)
    total_logps = (per_token_logps * target.target_mask).sum(dim=-1)  # [Batch, 1]
    assert target.target_mask is not None
    average_logps = total_logps / target.target_mask.sum(-1)

    return total_logps, average_logps


register_metric_formatter(
    "chosen_logps", "Chosen Sequence Log Probabilities", 50, format_as_float
)
register_metric_formatter(
    "rejected_logps", "Rejected Sequence Log Probabilities", 50, format_as_float
)
register_metric_formatter(
    "chosen_lengths", "Chosen Sequence Length", 70, format_as_float
)
register_metric_formatter(
    "rejected_lengths", "Rejected Sequence Length", 70, format_as_float
)


class PreferenceFinetuneMetricBag(SequenceMetricBag):
    """Holds the metrics of a sequence model preference finetuning task."""

    chosen_logps: Mean
    rejected_logps: Mean
    chosen_lengths: Mean
    rejected_lengths: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("chosen_logps", Mean(device=gang.device), persistent=False)
        self.register_metric(
            "rejected_logps", Mean(device=gang.device), persistent=False
        )
        self.register_metric(
            "chosen_lengths", Mean(device=gang.device), persistent=False
        )
        self.register_metric(
            "rejected_lengths", Mean(device=gang.device), persistent=False
        )

    @torch.inference_mode()
    def update_logps(
        self,
        batch: PreferenceOptimizationBatch,
        chosen_logps: Tensor,
        rejected_logps: Tensor,
    ) -> None:
        """Update the Chosen Sequence Log Probabilities and Rejected Sequence Log Probabilities metrics.

        :param batch:
            The batch processed by the model.
        :param chosen_logps:
            The log probabilities for each sequence in ``batch.chosen``.
        :param rejected_logps:
            The log probabilities for each sequence in ``batch.rejected``.
        """
        self.chosen_logps.update(
            chosen_logps.sum() / batch.chosen.batch_size, weight=batch.chosen.batch_size
        )
        self.rejected_logps.update(
            rejected_logps.sum() / batch.rejected.batch_size,
            weight=batch.rejected.batch_size,
        )

    @torch.inference_mode()
    def update_sequence_lengths(
        self,
        batch: PreferenceOptimizationBatch,
    ) -> None:
        """Update the Chosen Sequence Length and Rejected Sequence Length metrics.

        :param batch:
            The batch processed by the model.
        """
        self.chosen_lengths.update(
            Tensor([batch.chosen.num_target_elements() / batch.chosen.batch_size]),
            weight=batch.chosen.batch_size,
        )
        self.rejected_lengths.update(
            Tensor([batch.rejected.num_target_elements() / batch.rejected.batch_size]),
            weight=batch.rejected.batch_size,
        )


preference_unit_factories = ConfigBoundFactoryRegistry[
    [Module, Gang, Mapping[str, Gang]], TrainUnit[PreferenceOptimizationBatch]
]()

preference_unit_factory = preference_unit_factories.decorator
