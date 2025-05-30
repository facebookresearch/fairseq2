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
from fairseq2.metrics import Mean, MetricBag, Sum
from fairseq2.models.wav2vec2 import Wav2Vec2Loss, Wav2Vec2VectorQuantizerOutput


@torch.inference_mode()
def update_wav2vec2_loss(
    metric_bag: MetricBag, loss: Wav2Vec2Loss, num_targets: int
) -> None:
    n = num_targets

    d = num_targets * math.log(2)

    metric_bag.get(Mean, "loss").update(loss.aggregate.detach() / d, weight=n)

    metric_bag.get(Mean, "contrastive_loss").update(
        loss.contrastive.detach() / d, weight=n
    )

    metric_bag.get(Mean, "diversity_loss").update(loss.diversity.detach() / d, weight=n)

    metric_bag.get(Mean, "features_penalty").update(
        loss.features_penalty.detach() / d, weight=n
    )


@torch.inference_mode()
def update_wav2vec2_accuracy(metric_bag: MetricBag, logits: Tensor) -> None:
    # (N x S)
    predictions = logits.argmax(-1).view(-1)

    # wav2vec2 treats logit at index 0 as the target.
    targets = torch.zeros_like(predictions)

    metric_bag.get(MulticlassAccuracy, "accuracy").update(predictions, targets)


@torch.inference_mode()
def update_wav2vec2_quantizer_metrics(
    metric_bag: MetricBag, output: Wav2Vec2VectorQuantizerOutput
) -> None:
    metric_bag.get(Mean, "code_perplexity").update(output.code_perplexity)
    metric_bag.get(Mean, "prob_perplexity").update(output.prob_perplexity)

    metric_bag.get(Mean, "temperature").update(output.temperature)


@torch.inference_mode()
def update_wav2vec2_batch_metrics(metric_bag: MetricBag, batch: SequenceBatch) -> None:
    metric_bag.get(Sum, "num_examples").update(batch.num_examples)

    metric_bag.get(Sum, "num_elements").update(batch.num_elements)

    metric_bag.get(Sum, "total_num_examples").update(batch.num_examples)

    metric_bag.get(Sum, "total_num_elements").update(batch.num_elements)
