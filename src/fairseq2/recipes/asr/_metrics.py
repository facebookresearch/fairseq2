# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

from torch import Tensor

from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import Mean, MetricBag, Sum


def update_ctc_loss(metric_bag: MetricBag, loss: Tensor, batch_size: int) -> None:
    n = batch_size

    metric_bag.get(Mean, "ctc_loss").update(loss.detach() / n / math.log(2), weight=n)


def update_asr_batch_metrics(metric_bag: MetricBag, batch: Seq2SeqBatch) -> None:
    metric_bag.get(Sum, "num_examples").update(batch.num_examples)

    metric_bag.get(Sum, "num_elements").update(batch.num_elements)

    metric_bag.get(Sum, "total_num_examples").update(batch.num_examples)

    metric_bag.get(Sum, "total_num_elements").update(batch.num_elements)

    metric_bag.get(Sum, "padding").update(batch.padding)
