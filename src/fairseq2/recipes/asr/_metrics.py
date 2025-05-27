# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.datasets import Seq2SeqBatch
from fairseq2.device import Device
from fairseq2.metrics import Mean, MetricBag, Sum
from fairseq2.metrics.text import WerMetric


class AsrMetricBag(MetricBag):
    ctc_loss: Mean
    wer: WerMetric
    num_examples: Sum
    num_elements: Sum
    total_num_examples: Sum
    total_num_elements: Sum
    padding: Sum

    def __init__(self, device: Device) -> None:
        super().__init__()

        self.ctc_loss = Mean(device=device)

        self.wer = WerMetric(device=device)

        self.num_examples = Sum(device=device)

        self.num_elements = Sum(device=device)

        self.total_num_examples = Sum(device=device)

        self.total_num_elements = Sum(device=device)

        self.padding = Sum(device=device)

    @torch.inference_mode()
    def update_ctc_loss(self, batch: Seq2SeqBatch, loss: Tensor) -> None:
        n = batch.batch_size

        self.ctc_loss.update(loss.detach() / n / math.log(2), weight=n)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: Seq2SeqBatch) -> None:
        self.num_examples.update(batch.num_examples)

        self.num_elements.update(batch.num_elements)

        self.total_num_examples.update(batch.num_examples)

        self.total_num_elements.update(batch.num_elements)

        self.padding.update(batch.padding)

    @override
    def process_metric_values(self, values: dict[str, object]) -> None:
        super().process_metric_values(values)

        value = values.pop("wer")

        if isinstance(value, tuple):
            uer, wer = value

            if isinstance(uer, Tensor) and isinstance(wer, Tensor):
                if uer >= 1.0 and wer >= 1.0:
                    values["uer"] = uer
                    values["wer"] = wer
