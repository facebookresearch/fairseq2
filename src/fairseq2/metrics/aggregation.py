# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch import Tensor
from torcheval.metrics import Max as MaxBase
from torcheval.metrics import Mean as MeanBase
from torcheval.metrics import Min as MinBase
from torcheval.metrics import Sum as SumBase
from typing_extensions import Self, override

from fairseq2.device import CPU
from fairseq2.utils.tensor import to_tensor


class Min(MinBase):
    """See :class:`MinBase`."""

    @override
    def update(self, input_: int | float | Tensor) -> Self:
        if isinstance(input_, (int, float)):
            input_ = to_tensor(input_, device=CPU)

        super().update(input_)

        return self


class Max(MaxBase):
    """See :class:`MaxBase`."""

    @override
    def update(self, input_: int | float | Tensor) -> Self:
        if isinstance(input_, (int, float)):
            input_ = to_tensor(input_, device=CPU)

        super().update(input_)

        return self


class Mean(MeanBase):
    """See :class:`MeanBase`."""

    @override
    def update(
        self,
        input_: int | float | Tensor,
        *,
        weight: int | float | Tensor = 1.0,
    ) -> Self:
        if isinstance(input_, (int, float)):
            input_ = to_tensor(input_, device=CPU)

        super().update(input_, weight=weight)

        return self


class Sum(SumBase):
    """See :class:`SumBase`."""

    @override
    def update(
        self,
        input_: int | float | Tensor,
        *,
        weight: int | float | Tensor = 1.0,
    ) -> Self:
        if isinstance(input_, (int, float)):
            input_ = to_tensor(input_, device=CPU)

        super().update(input_, weight=weight)

        return self
