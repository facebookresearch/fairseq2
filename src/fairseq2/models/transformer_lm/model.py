# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Literal, final, overload

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.clm import CausalLM
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.models.transformer_lm.decoder import TransformerLMDecoder
from fairseq2.nn import BatchLayout, IncrementalStateBag, Projection
from fairseq2.nn.functional import cross_entropy


@final
class TransformerLM(CausalLM):
    """Represents a decoder-only Transformer model."""

    def __init__(
        self,
        model_dim: int,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerLMDecoder,
        final_proj: Projection,
        pad_idx: int | None,
        max_seq_len: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param decoder_frontend: The decoder frontend.
        :param decoder: The decoder.
        :param final_proj: The projection to apply to decoder outputs.
        :param max_seq_len: The maximum length of produced sequences.
        """
        super().__init__(max_seq_len)

        self.model_dim = model_dim
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj
        self.pad_idx = pad_idx

    @override
    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = ...,
    ) -> Tensor: ...

    @override
    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
    ) -> Tensor: ...

    @override
    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: Literal[False],
    ) -> Tensor: ...

    @override
    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    @override
    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: bool = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        seqs, seqs_layout = self.decoder_frontend(
            seqs, seqs_layout, state_bag=state_bag
        )

        decoder_output = self.decoder(seqs, seqs_layout, state_bag=state_bag)

        del seqs

        if targets is None:
            return self.final_proj(decoder_output)

        if not return_logits:
            return self.compute_fused_loss(
                decoder_output,
                targets,
                label_smoothing=label_smoothing,
                target_mask=target_mask,
                reduction=reduction,
            )

        logits = self.final_proj(decoder_output)

        del decoder_output

        loss = self.compute_loss(
            logits,
            targets,
            label_smoothing=label_smoothing,
            target_mask=target_mask,
            reduction=reduction,
        )

        return loss, logits

    def compute_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
    ) -> Tensor:
        return cross_entropy(
            logits,
            targets,
            self.pad_idx,
            label_smoothing=label_smoothing,
            target_mask=target_mask,
            reduction=reduction,
        )

    def compute_fused_loss(
        self,
        decoder_output: Tensor,
        targets: Tensor,
        *,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
    ) -> Tensor:
        logits = self.final_proj(decoder_output)

        return cross_entropy(
            logits,
            targets,
            self.pad_idx,
            label_smoothing=label_smoothing,
            target_mask=target_mask,
            reduction=reduction,
        )

    def compile_loss(self, *args: Any, **kwargs: Any) -> None:
        self.compute_fused_loss = torch.compile(  # type: ignore[method-assign]
            self.compute_fused_loss, *args, **kwargs
        )

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"pad_idx={self.pad_idx}, "
            f"max_seq_len={self.max_seq_len}"
        )
