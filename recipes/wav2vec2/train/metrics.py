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
def update_wav2vec2_loss(
    metric_bag: MetricBag, loss: Wav2Vec2Loss, num_targets: int
) -> None:
    """Update wav2vec2 loss metrics.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/recipes/wav2vec2/_common.py:121-132
    Method: Wav2Vec2MetricBag.update_losses()
    """
    n = num_targets
    d = num_targets * math.log(2)

    metric_bag.get(Mean, "loss").update(
        loss.aggregate.detach() / d, weight=n
    )  # ORIGINAL: line 126
    metric_bag.get(Mean, "contrastive_loss").update(
        loss.contrastive.detach() / d, weight=n  # ORIGINAL: line 128
    )
    metric_bag.get(Mean, "diversity_loss").update(
        loss.diversity.detach() / d, weight=n
    )  # ORIGINAL: line 130
    metric_bag.get(Mean, "feature_penalty").update(
        loss.features_penalty.detach() / d, weight=n  # ORIGINAL: line 132
    )


@torch.inference_mode()
def update_wav2vec2_accuracy(metric_bag: MetricBag, output: Wav2Vec2Output) -> None:
    """Update wav2vec2 accuracy metrics.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/recipes/wav2vec2/_common.py:135-142
    Method: Wav2Vec2MetricBag.update_accuracy()
    """
    # (N x S) - ORIGINAL: line 136
    predictions = output.logits.argmax(-1).view(-1)  # ORIGINAL: line 137

    # wav2vec2 treats logit at index 0 as the target. - ORIGINAL: line 139
    targets = torch.zeros_like(predictions)  # ORIGINAL: line 140

    metric_bag.get(MulticlassAccuracy, "accuracy").update(
        predictions, targets
    )  # ORIGINAL: line 142


@torch.inference_mode()
def update_wav2vec2_quantizer_metrics(
    metric_bag: MetricBag, output: Wav2Vec2VectorQuantizerOutput
) -> None:
    """Update wav2vec2 quantizer metrics.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/recipes/wav2vec2/_common.py:145-152
    Method: Wav2Vec2MetricBag.update_quantizer_metrics()
    """
    # ORIGINAL: line 146-147 - Only update for GumbelVectorQuantizerOutput
    if not isinstance(output, GumbelWav2Vec2VectorQuantizer):
        return

    metric_bag.get(Mean, "code_perplexity").update(
        output.code_perplexity
    )  # ORIGINAL: line 149
    metric_bag.get(Mean, "prob_perplexity").update(
        output.prob_perplexity
    )  # ORIGINAL: line 150
    metric_bag.get(Mean, "temperature").update(output.temperature)  # ORIGINAL: line 152


@torch.inference_mode()
def update_wav2vec2_batch_metrics(metric_bag: MetricBag, batch: SequenceBatch) -> None:
    """Update batch metrics.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/recipes/wav2vec2/_common.py:155-169
    Method: Wav2Vec2MetricBag.update_batch_metrics()
    """
    num_examples = batch.batch_size  # ORIGINAL: line 157
    num_elements = batch.num_elements  # ORIGINAL: line 159

    metric_bag.get(Sum, "num_examples").update(num_examples)  # ORIGINAL: line 161
    metric_bag.get(Sum, "num_elements").update(num_elements)  # ORIGINAL: line 162

    # Training-specific metrics - ORIGINAL: lines 164-169
    # Note: In v0.5 we assume training mode since this is the train recipe
    metric_bag.get(Sum, "total_num_examples").update(num_examples)  # ORIGINAL: line 168
    metric_bag.get(Sum, "total_num_elements").update(num_elements)  # ORIGINAL: line 169
