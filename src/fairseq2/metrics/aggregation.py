# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
from torch import Tensor
from torcheval.metrics import Max as MaxBase
from torcheval.metrics import Mean as MeanBase
from torcheval.metrics import Metric
from torcheval.metrics import Min as MinBase
from torcheval.metrics import Sum as SumBase
from typing_extensions import Self

from fairseq2.typing import Device, override


class Min(MinBase):
    """See :class:`MinBase`."""

    @override
    def update(self, input_: Union[int, float, Tensor]) -> Self:
        if isinstance(input_, (int, float)):
            input_ = torch.tensor(input_)

        super().update(input_)

        return self


class Max(MaxBase):
    """See :class:`MaxBase`."""

    @override
    def update(self, input_: Union[int, float, Tensor]) -> Self:
        if isinstance(input_, (int, float)):
            input_ = torch.tensor(input_)

        super().update(input_)

        return self


class Mean(MeanBase):
    """See :class:`MeanBase`."""

    @override
    def update(
        self,
        input_: Union[int, float, Tensor],
        *,
        weight: Union[int, float, Tensor] = 1.0,
    ) -> Self:
        if isinstance(input_, (int, float)):
            input_ = torch.tensor(input_)

        super().update(input_, weight=weight)

        return self


class Sum(SumBase):
    """See :class:`SumBase`."""

    @override
    def update(
        self,
        input_: Union[int, float, Tensor],
        *,
        weight: Union[int, float, Tensor] = 1.0,
    ) -> Self:
        if isinstance(input_, (int, float)):
            input_ = torch.tensor(input_)

        super().update(input_, weight=weight)

        return self


class MaxSum(Metric[Tensor]):
    """Calculate the sum of all elements in all the input tensors locally and
    take the maximum value when merged with other metrics."""

    sum_: Tensor

    def __init__(self, *, device: Optional[Device] = None) -> None:
        super().__init__(device=device)

        sum_ = torch.zeros((), device=device, dtype=torch.int64)

        self._add_state("sum_", sum_)

    @override
    @torch.inference_mode()
    def update(self, input_: Union[int, Tensor]) -> Self:
        self.sum_ += input_

        return self

    @override
    @torch.inference_mode()
    def compute(self) -> Tensor:
        return self.sum_

    @override
    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[MaxSum]) -> Self:
        for metric in metrics:
            self.sum_ = torch.max(self.sum_, metric.sum_.to(self.device))

        return self
