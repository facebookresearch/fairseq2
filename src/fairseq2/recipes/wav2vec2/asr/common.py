# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from torch import Tensor

from fairseq2.gang import Gang
from fairseq2.metrics.aggregation import Mean
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.recipes.common_metrics import TaskMetricBag


class Wav2Vec2AsrMetricBag(TaskMetricBag):
    """Holds the metrics of a wav2vec 2.0 ASR model training or evaluation task."""

    _ctc_loss: Mean

    def __init__(self, gang: Gang, train: bool = True) -> None:
        super().__init__(gang, train=train)

        d = gang.device

        self.register_metric("_ctc_loss", Mean(device=d), persistent=False)

    @torch.inference_mode()
    def update_ctc_loss(self, batch: Seq2SeqBatch, loss: Tensor) -> None:
        """Update the CTC loss metric.

        :param batch:
            The batch processed by the model.
        :param ctc_loss:
            The loss of ``batch``.
        """
        normalized_loss = loss.detach() / batch.batch_size / math.log(2)

        self._ctc_loss.update(normalized_loss, weight=batch.batch_size)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: Seq2SeqBatch) -> None:
        """Update the batch metrics.

        :param batch:
            The batch processed by the model.
        """
        num_examples = batch.batch_size
        num_elements = batch.num_source_elements()

        self._num_batches.update(1)

        self._num_examples.update(num_examples)
        self._num_elements.update(num_elements)

        if self._train:
            assert self._total_num_examples is not None
            assert self._total_num_elements is not None

            self._total_num_examples.update(num_examples)
            self._total_num_elements.update(num_elements)
