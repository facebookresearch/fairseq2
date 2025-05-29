# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from torch import Tensor
from torcheval.metrics import MulticlassAccuracy

from fairseq2.datasets import SequenceBatch
from fairseq2.device import Device
from fairseq2.metrics import Mean, MetricBag, Sum
from fairseq2.models.wav2vec2 import Wav2Vec2Loss, Wav2Vec2VectorQuantizerOutput


class Wav2Vec2MetricBag(MetricBag):
    loss: Mean
    contrastive_loss: Mean
    diversity_loss: Mean
    features_penalty: Mean
    accuracy: MulticlassAccuracy
    code_perplexity: Mean
    prob_perplexity: Mean
    temperature: Mean
    num_examples: Sum
    num_elements: Sum
    total_num_examples: Sum
    total_num_elements: Sum

    def __init__(self, device: Device) -> None:
        super().__init__()

        self.loss = Mean(device=device)

        self.contrastive_loss = Mean(device=device)

        self.diversity_loss = Mean(device=device)

        self.features_penalty = Mean(device=device)

        self.accuracy = MulticlassAccuracy(device=device)

        self.code_perplexity = Mean(device=device)

        self.prob_perplexity = Mean(device=device)

        self.temperature = Mean(device=device)

        self.num_examples = Sum(device=device)

        self.num_elements = Sum(device=device)

        self.total_num_examples = Sum(device=device)

        self.total_num_elements = Sum(device=device)

    @torch.inference_mode()
    def update_loss(self, loss: Wav2Vec2Loss, num_targets: int) -> None:
        n = num_targets

        d = num_targets * math.log(2)

        self.loss.update(loss.aggregate.detach() / d, weight=n)

        self.contrastive_loss.update(loss.contrastive.detach() / d, weight=n)

        self.diversity_loss.update(loss.diversity.detach() / d, weight=n)

        self.features_penalty.update(loss.features_penalty.detach() / d, weight=n)

    @torch.inference_mode()
    def update_accuracy(self, logits: Tensor) -> None:
        # (N x S)
        predictions = logits.argmax(-1).view(-1)

        # wav2vec2 treats logit at index 0 as the target.
        targets = torch.zeros_like(predictions)

        self.accuracy.update(predictions, targets)

    @torch.inference_mode()
    def update_quantizer_metrics(self, output: Wav2Vec2VectorQuantizerOutput) -> None:
        self.code_perplexity.update(output.code_perplexity)
        self.prob_perplexity.update(output.prob_perplexity)

        self.temperature.update(output.temperature)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: SequenceBatch) -> None:
        self.num_examples.update(batch.num_examples)

        self.num_elements.update(batch.num_elements)

        self.total_num_examples.update(batch.num_examples)

        self.total_num_elements.update(batch.num_elements)
