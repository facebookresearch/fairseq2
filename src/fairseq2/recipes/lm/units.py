# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, final

from torch import Tensor
from torch.nn import Module

from fairseq2.gang import Gang
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModel,
    SequenceModelOutput,
    as_auto_regressive_input,
)
from fairseq2.recipes.metrics import SequenceModelMetricBag
from fairseq2.recipes.trainer import AbstractTrainUnit
from fairseq2.recipes.utils.setup import check_model_type
from fairseq2.typing import override


@final
class InstructionTrainUnit(AbstractTrainUnit[SequenceBatch]):
    """Represents the training unit of an instruction model."""

    _metric_bag: SequenceModelMetricBag

    def __init__(self, model: Module, gang: Gang) -> None:
        """
        :param model:
            The instruction model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed training.
        """
        super().__init__(model)

        check_model_type(model, SequenceModel)

        self._metric_bag = SequenceModelMetricBag(gang)

    @override
    def __call__(self, batch: SequenceBatch) -> Tuple[Tensor, int]:
        input_batch, target_batch = as_auto_regressive_input(batch)

        output = self._forward(input_batch)

        loss = output.compute_loss(
            target_batch.seqs, loss_mask=target_batch.target_mask
        )

        self._metric_bag.update_loss_metrics(target_batch, loss.detach())

        return loss, target_batch.num_target_elements()

    def _forward(self, batch: SequenceBatch) -> SequenceModelOutput:
        return self._model(batch)  # type: ignore[no-any-return]

    @property
    @override
    def metric_bag(self) -> SequenceModelMetricBag:
        return self._metric_bag
