# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from torcheval.metrics import MulticlassAccuracy

from fairseq2.datasets import SequenceBatch
from fairseq2.metrics import Mean, MetricBag, Sum
from fairseq2.models.wav2vec2 import (
    GumbelWav2Vec2VectorQuantizer,
    Wav2Vec2Loss,
    Wav2Vec2Output,
    Wav2Vec2VectorQuantizerOutput,
)

@torch.inference_mode()
def add_ssl_metrics(metric_bag: MetricBag) -> None:
    metric_bag.add("loss", Mean())
    metric_bag.add("accuracy", MulticlassAccuracy())
    metric_bag.add("contrastive_loss", Mean())
    metric_bag.add("diversity_loss", Mean())
    metric_bag.add("feature_penalty", Mean())
    metric_bag.add("code_perplexity", Mean())
    metric_bag.add("prob_perplexity", Mean())
    metric_bag.add("temperature", Mean())
    metric_bag.add("num_examples", Sum())
    metric_bag.add("num_elements", Sum())
    metric_bag.add("total_num_examples", Sum())
    metric_bag.add("total_num_elements", Sum())


@torch.inference_mode()
def update_wav2vec2_loss(
    metric_bag: MetricBag, loss: Wav2Vec2Loss, num_targets: int
) -> None:
    """Update wav2vec2 loss metrics."""
    n = num_targets
    d = num_targets * math.log(2)

    metric_bag.get(Mean, "loss").update(loss.aggregate.detach() / d, weight=n)
    metric_bag.get(Mean, "contrastive_loss").update(
        loss.contrastive.detach() / d, weight=n
    )
    metric_bag.get(Mean, "diversity_loss").update(loss.diversity.detach() / d, weight=n)
    metric_bag.get(Mean, "feature_penalty").update(
        loss.features_penalty.detach() / d, weight=n
    )


@torch.inference_mode()
def update_wav2vec2_accuracy(metric_bag: MetricBag, output: Wav2Vec2Output) -> None:
    """Update wav2vec2 accuracy metrics."""
    # (N x S)
    predictions = output.logits.argmax(-1).view(-1)

    # wav2vec2 treats logit at index 0 as the target.
    targets = torch.zeros_like(predictions)

    metric_bag.get(MulticlassAccuracy, "accuracy").update(predictions, targets)


@torch.inference_mode()
def update_wav2vec2_quantizer_metrics(
    metric_bag: MetricBag, output: Wav2Vec2VectorQuantizerOutput
) -> None:
    """Update wav2vec2 quantizer metrics."""
    if not isinstance(output, GumbelWav2Vec2VectorQuantizer):
        return

    metric_bag.get("code_perplexity", Mean).update(output.code_perplexity)
    metric_bag.get("prob_perplexity", Mean).update(output.prob_perplexity)
    metric_bag.get("temperature", Mean).update(output.temperature)


@torch.inference_mode()
def update_wav2vec2_batch_metrics(metric_bag: MetricBag, batch: SequenceBatch) -> None:
    """Update batch metrics."""
    num_examples = batch.batch_size
    num_elements = batch.num_elements

    metric_bag.get("total_num_examples", Sum).update(num_examples)
    metric_bag.get("total_num_elements", Sum).update(num_elements)
