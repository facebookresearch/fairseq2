# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from typing_extensions import override

from fairseq2.error import NotSupportedError
from fairseq2.nn import BatchLayout

# isort: split

from fairseq2.models.transformer._attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    CausalAttentionBias,
    maybe_get_attention_bias_tensor,
)
from fairseq2.models.transformer._sdpa._base import SDPA


@final
class TorchSDPA(SDPA):
    """Computes scaled dot-product attention using PyTorch SDPA."""

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
        *,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if needs_weights:
            raise NotSupportedError(f"`{TorchSDPA}` does not support `needs_weights`.")

        is_causal = False

        # ([[N], H], S, S_kv)
        if isinstance(self.bias, CausalAttentionBias):
            if self.bias.attn_window_len is None:
                full_seqs = not seqs_layout.packed and not seqs_layout.padded
                full_keys = not keys_layout.packed and not keys_layout.padded

                if full_seqs and full_keys:
                    seq_len = seqs.size(1)
                    key_len = keys.size(1)

                    is_causal = seq_len == key_len

        if is_causal:
            bias = None
        else:
            # ([[N], H], S, S_kv)
            bias = maybe_get_attention_bias_tensor(
                self.bias, seqs, seqs_layout, keys_layout, bias_cache
            )

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout_p

        q, k, v = seqs, keys, values

        # (N, S, H, K) -> (N, H, S, K)
        q = q.transpose(-2, -3)

        # (N, S_kv, H, K) -> (N, H, S_kv, K)
        k = k.transpose(-2, -3)

        # (N, S_kv, H, V) -> (N, H, S_kv, V)
        v = v.transpose(-2, -3)

        # (N, H, S, V)
        attns = scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=dropout_p, is_causal=is_causal
        )

        # (N, H, S, V) -> (N, S, H, V)
        attns = attns.transpose(-2, -3)

        return attns, None

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"bias={self.bias}, dropout_p={self.dropout_p:G}"
