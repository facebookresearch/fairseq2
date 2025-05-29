# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from torch.nn import Module

from fairseq2.datasets import SequenceBatch

# isort: split

from fairseq2.recipes.wav2vec2._metrics import Wav2Vec2MetricBag


@final
class Wav2Vec2Criterion:
    _module: Module
    _diversity_weight: float
    _features_penalty_weight: float

    def __init__(
        self, module: Module, diversity_weight: float, features_penalty_weight: float
    ) -> None:
        self._module = module

        self._diversity_weight = diversity_weight

        self._features_penalty_weight = features_penalty_weight

    def __call__(
        self, batch: SequenceBatch, metric_bag: Wav2Vec2MetricBag
    ) -> tuple[Tensor, int]:
        seqs, seqs_layout = batch.as_input()

        w2v2_loss, w2v2_output = self._module(
            seqs,
            seqs_layout,
            diversity_weight=self._diversity_weight,
            features_penalty_weight=self._features_penalty_weight,
        )

        metric_bag.update_loss(w2v2_loss, w2v2_output.num_targets)

        metric_bag.update_accuracy(w2v2_output.logits)

        metric_bag.update_quantizer_metrics(w2v2_output.quantizer_output)

        metric_bag.update_batch_metrics(batch)

        return w2v2_loss.aggregate, w2v2_output.num_targets
