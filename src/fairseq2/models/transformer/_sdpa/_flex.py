# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable, TypeAlias, final

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention
from typing_extensions import override

from fairseq2.models.transformer._block_mask import BlockMaskCache
from fairseq2.nn import BatchLayout

# isort: split

from fairseq2.models.transformer._attention_bias import (
    AttentionBias,
    AttentionBiasCache,
)
from fairseq2.models.transformer._sdpa._base import SDPA

MaskFunction: TypeAlias = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]

flex_attention = torch.compile(flex_attention, dynamic=False)


@final
class FlexSDPA(SDPA):
    """Computes scaled dot-product attention using PyTorch's Flex Attention."""

    bias: AttentionBias
    dropout_p: float

    def __init__(self, bias: AttentionBias, *, dropout_p: float = 0.0) -> None:
        super().__init__()

        self.bias = bias
        self.dropout_p = dropout_p

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        keys: Tensor,
        keys_layout: BatchLayout,
        values: Tensor,
        bias_cache: AttentionBiasCache,
        block_mask_cache: BlockMaskCache,
        *,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if seqs_layout.packed ^ keys_layout.packed:
            raise ValueError("`seqs_layout` and `keys_layout` must be both packed.")

        unsqueezed = False
        if seqs.ndim == 3:
            unsqueezed = True
            seqs = seqs.unsqueeze(0)
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)

        # Create the composed block mask using and_masks
        block_mask = block_mask_cache.get_or_create_mask(
            self.bias,
            seqs_layout,
            keys_layout,
            seqs.device,
        )

        seqs = seqs.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attns = flex_attention(
            seqs,
            keys,
            values,
            block_mask=block_mask,
            enable_gqa=False,
        )

        if isinstance(attns, tuple):
            attns, _ = attns

        attns = attns.transpose(1, 2)
        if unsqueezed:
            attns = attns.squeeze(0)

        return attns, None

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"bias={self.bias}, dropout_p={self.dropout_p:G}"
