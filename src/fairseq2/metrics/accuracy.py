# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Iterable, Optional, Union, final

import torch
from torch import Tensor
from torcheval.metrics import Metric
from typing_extensions import Self

from fairseq2.typing import Device, override


@final
class AccuracyMetric(Metric[Tensor]):
    """Computes the Accuracy metric."""

    correct_preds: Tensor
    total_preds: Tensor

    def __init__(self, *, device: Optional[Device] = None) -> None:
        super().__init__(device=device)

        self._add_state(
            "correct_preds", torch.zeros((), device=device, dtype=torch.float64)
        )
        self._add_state(
            "total_preds", torch.zeros((), device=device, dtype=torch.float64)
        )

    @override
    @torch.inference_mode()
    def update(
        self, correct_preds: Union[int, Tensor], total_preds: Union[int, Tensor]
    ) -> Self:
        """
        :param correct_preds:
            The number of correct predictions.
        :param total_preds:
            The number of total predictions.
        """
        self.correct_preds += correct_preds
        self.total_preds += total_preds

        return self

    @override
    @torch.inference_mode()
    def compute(self) -> Tensor:
        if self.total_preds:
            accuracy = self.correct_preds / self.total_preds
        else:
            accuracy = torch.zeros((), dtype=torch.float64)

        return accuracy

    @override
    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[AccuracyMetric]) -> Self:
        for metric in metrics:
            self.correct_preds += metric.correct_preds.to(self.device)
            self.total_preds += metric.total_preds.to(self.device)

        return self
