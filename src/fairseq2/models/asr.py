# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, overload

from torch import Tensor
from torch.nn import Module

from fairseq2.nn import BatchLayout


class AsrModel(Module, ABC):
    """Represents a CTC-based Automatic Speech Recognition model."""

    @overload
    def forward(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout]: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        targets_layout: BatchLayout,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        targets_layout: BatchLayout,
        *,
        return_logits: Literal[False],
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        targets_layout: BatchLayout,
        *,
        return_logits: Literal[True],
    ) -> tuple[Tensor, Tensor, BatchLayout]: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        targets_layout: BatchLayout,
        *,
        return_logits: bool = ...,
    ) -> Tensor | tuple[Tensor, Tensor, BatchLayout]: ...

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor | None = None,
        targets_layout: BatchLayout | None = None,
        *,
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, BatchLayout] | tuple[Tensor, Tensor, BatchLayout]: ...

    if TYPE_CHECKING:
        __call__ = forward
