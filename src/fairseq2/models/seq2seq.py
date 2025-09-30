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

from fairseq2.nn import BatchLayout, IncrementalStateBag


class Seq2SeqModel(Module, ABC):
    """Represents a sequence-to-sequence model."""

    def __init__(self, max_source_seq_len: int, max_target_seq_len: int) -> None:
        super().__init__()

        self.max_source_seq_len = max_source_seq_len
        self.max_target_seq_len = max_target_seq_len

    @overload
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = ...,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: Literal[False],
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: bool = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...

    @abstractmethod
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        targets: Tensor | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]: ...

    if TYPE_CHECKING:
        __call__ = forward
