# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import final

import torch
from torch import Tensor
from torch.nn import Module
from torcheval.metrics import MulticlassAccuracy

from fairseq2.gang import Gang
from fairseq2.metrics.aggregation import Mean
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.bestrq import BestRQLoss, BestRQModel, BestRQOutput
from fairseq2.models.bestrq import MultiRandomVectorQuantizerOutput
from fairseq2.recipes.common_metrics import BaseMetricBag
from fairseq2.recipes.utils.setup import check_model_type


@final
class BestRQCriterion:
    _model: Module
    _label_smoothing: float

    def __init__(
        self, model: Module, label_smoothing: float
    ) -> None:
        check_model_type(model, BestRQModel)

        self._model = model

        self._label_smoothing = label_smoothing

    def __call__(
        self, batch: SequenceBatch, metric_bag: BestRQMetricBag
    ) -> tuple[Tensor, int]:
        output = self._forward(batch)

        loss = output.compute_loss(label_smoothing=self._label_smoothing)

        batch_size, seq_len = output.logits.shape[:2]

        num_targets = batch_size * seq_len

        metric_bag.update_losses(loss, num_targets)

        metric_bag.update_accuracy(output)

        metric_bag.update_entropy_metrics(output)

        metric_bag.update_batch_metrics(batch)

        return loss.total, num_targets

    def _forward(self, batch: SequenceBatch) -> BestRQOutput:
        return self._model(batch)  # type: ignore[no-any-return]

    @property
    def model(self) -> Module:
        return self._model


class BestRQMetricBag(BaseMetricBag):
    loss: Mean
    contrastive_loss: Mean
    diversity_loss: Mean
    feature_penalty: Mean
    accuracy: MulticlassAccuracy
    quantizer_entropy: Mean
    encoder_entropy: Mean
    temperature: Mean

    def __init__(self, gang: Gang, train: bool = True) -> None:
        super().__init__(gang, train=train)

        d = gang.device

        self.register_metric("loss", Mean(device=d), persistent=False)

        self.register_metric("bert_loss", Mean(device=d), persistent=False)
        
        self.register_metric("accuracy", MulticlassAccuracy(device=d), persistent=False)

        self.register_metric("quantizer_entropy", Mean(device=d), persistent=False)
        self.register_metric("encoder_entropy", Mean(device=d), persistent=False)
        
    @torch.inference_mode()
    def update_losses(self, loss: BestRQLoss, num_targets: int) -> None:
        n = num_targets

        d = num_targets * math.log(2)

        self.loss.update(loss.total.detach() / d, weight=n)

        self.bert_loss.update(loss.bert.detach() / d, weight=n)

    @torch.inference_mode()
    def update_accuracy(self, output: BestRQOutput) -> None:
        # (N x S)
        predictions = output.logits.argmax(-1)

        for pred, trgt in zip(predictions, output.targets):
            self.accuracy.update(pred, trgt)

    @torch.inference_mode()
    def update_entropy_metrics(self, output: BestRQOutput) -> None:
        if not isinstance(output, BestRQOutput):
            return
        
        self.quantizer_entropy.update(output.quantizer_output.compute_target_entropy())
        self.encoder_entropy.update(output.compute_encoder_entropy())
        
    @torch.inference_mode()
    def update_batch_metrics(self, batch: SequenceBatch) -> None:
        """Update the batch metrics."""
        num_examples = batch.batch_size

        num_elements = batch.num_elements()

        self.num_examples.update(num_examples)
        self.num_elements.update(num_elements)

        if self._train:
            assert self.total_num_examples is not None
            assert self.total_num_elements is not None

            self.total_num_examples.update(num_examples)
            self.total_num_elements.update(num_elements)
