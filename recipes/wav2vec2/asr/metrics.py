# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from torch import Tensor

from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import Mean, MetricBag, Sum


@torch.inference_mode()
def update_ctc_loss(metric_bag: MetricBag, loss: Tensor, batch_size: int) -> None:
    """Update CTC loss metrics."""
    n = batch_size

    # Normalize by batch size and convert to bits (log base 2)
    metric_bag.get(Mean, "ctc_loss").update(loss.detach() / n / math.log(2), weight=n)


@torch.inference_mode()
def update_asr_batch_metrics(metric_bag: MetricBag, batch: Seq2SeqBatch) -> None:
    """Update ASR batch size and element count metrics."""
    metric_bag.get(Sum, "num_examples").update(batch.num_examples)
    metric_bag.get(Sum, "num_elements").update(batch.num_elements)
    metric_bag.get(Sum, "total_num_examples").update(batch.num_examples)
    metric_bag.get(Sum, "total_num_elements").update(batch.num_elements)
    metric_bag.get(Sum, "padding").update(batch.padding)
