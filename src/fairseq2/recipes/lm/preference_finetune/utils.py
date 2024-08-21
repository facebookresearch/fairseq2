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

from fairseq2.gang import Gang
from fairseq2.logging import LogWriter
from fairseq2.metrics.recorder import format_as_float, register_metric_formatter
from fairseq2.models import load_model
from fairseq2.nn.utils.module import freeze_parameters
from fairseq2.recipes.common_metrics import SequenceMetricBag
from fairseq2.recipes.lm.preference_finetune.recipe import PreferenceOptimizationBatch
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

    _chosen_logps: Mean
    _rejected_logps: Mean
    _chosen_lengths: Mean
    _rejected_lengths: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric(
            "_chosen_logps", Mean(device=gang.device), persistent=False
        )
        self.register_metric(
            "_rejected_logps", Mean(device=gang.device), persistent=False
        )
        self.register_metric(
            "_chosen_lengths", Mean(device=gang.device), persistent=False
        )
        self.register_metric(
            "_rejected_lengths", Mean(device=gang.device), persistent=False
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
        self._chosen_logps.update(
            chosen_logps.sum() / batch.chosen.batch_size, weight=batch.chosen.batch_size
        )
        self._rejected_logps.update(
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
        if batch.chosen.target_mask is not None:
            chosen_lengths = batch.chosen.target_mask.sum(dim=-1)
        else:
            chosen_lengths = batch.chosen.seqs.count_nonzero(dim=-1)

        if batch.rejected.target_mask is not None:
            rejected_lengths = batch.rejected.target_mask.sum(dim=-1)
        else:
            rejected_lengths = batch.rejected.seqs.count_nonzero(dim=-1)

        self._chosen_lengths.update(
            chosen_lengths.sum() / batch.chosen.batch_size,
            weight=batch.chosen.batch_size,
        )
        self._rejected_lengths.update(
            rejected_lengths.sum() / batch.rejected.batch_size,
            weight=batch.rejected.batch_size,
        )
